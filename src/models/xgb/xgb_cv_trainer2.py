import glob
import gc
import os
from time import perf_counter as now
from pathlib import Path

import cudf
import cupy as cp
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import seaborn as sns
import shap
import wandb
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from src.utils.get_cat_cols import get_cat_cols
from src.utils.print_duration import print_duration

try:
    import cudf
    _HAS_CUDF = True
except Exception:
    _HAS_CUDF = False


class ParquetIter(xgb.core.DataIter):
    def __init__(
        self,
        paths,
        features=None,
        target="target",
        cat_cols=None,
        fold_col=None,
        include_folds=None,
        exclude_folds=None,
        weight_col=None,
        batch_rows=1_000_000,
        use_cudf=None,
        extra_exclude_cols=None,
        predict_mode=False,      # ← 追加: 推論モード（test 用）
        keep_row_ids=True,       # ← 追加: row_id を回収して後で取り出す
    ):
        super().__init__()
        self._temporary_data = None 
        if isinstance(paths, (str, os.PathLike)):
            paths = [str(paths)]
        self.paths = [str(p) for p in paths]
        self.target = target
        self.weight_col = weight_col
        self.fold_col = fold_col
        self.include_folds = None if include_folds is None else set(include_folds)
        self.exclude_folds = None if exclude_folds is None else set(exclude_folds)
        self.batch_rows = int(batch_rows)
        self.cat_cols = list(cat_cols or [])

        self.use_cudf = _HAS_CUDF if use_cudf is None else bool(use_cudf)

        self.predict_mode = bool(predict_mode)
        self.keep_row_ids = bool(keep_row_ids)
        self._row_id_chunks = []
        self._pass_count = 0

        # --- スキーマから特徴量列を自動決定（features=Noneのとき） ---
        schema = ds.dataset(self.paths, format="parquet").schema
        all_cols = [f.name for f in schema]

        if features is None:
            meta = {"row_id"}
            if (not self.predict_mode) and (self.target in all_cols):
                meta.add(self.target)
            if (not self.predict_mode) and self.weight_col:
                meta.add(self.weight_col)
            if (not self.predict_mode) and self.fold_col:
                meta.add(self.fold_col)
            if extra_exclude_cols:
                meta |= set(extra_exclude_cols)
            self.features = [c for c in all_cols if c not in meta]
        else:
            self.features = list(features)

        # 入力列（重複除去）
        cols = list(self.features)
        if (not self.predict_mode) and (self.target in all_cols):
            cols.append(self.target)
        if (not self.predict_mode) and (self.weight_col in all_cols):
            cols.append(self.weight_col)
        if (not self.predict_mode) and self.fold_col:
            cols.append(self.fold_col)
        cols.append("row_id")
        self._columns = list(dict.fromkeys(cols))

        # 内部状態
        self._reader = None
        self._current_file_index = 0

    # row_id を取り出すためのヘルパ
    def collected_row_ids(self):
        if not self._row_id_chunks:
            return None
        return np.concatenate(self._row_id_chunks).flatten()

    def reset(self):
        self._current_file_index = 0
        self._reader = None
        self._pass_count += 1 

    def next(self, input_data):
        while True:
            if self._reader is None:
                if not self._prepare_next_file():
                    return 0
            try:
                batch = next(self._reader)
            except StopIteration:
                self._reader = None
                continue

            if self.use_cudf:
                # RecordBatch -> Table -> cuDF
                if isinstance(batch, pa.RecordBatch):
                    batch = pa.Table.from_batches([batch])
                df = cudf.DataFrame.from_arrow(batch)
            else:
                df = batch.to_pandas()

            if self.cat_cols and not self.use_cudf:
                for c in self.cat_cols:
                    if c in df.columns:
                        df[c] = df[c].astype("category")

            if self.keep_row_ids and self._pass_count == 1 and "row_id" in df.columns:
                self._row_id_chunks.append(df["row_id"].to_numpy())

            if self.predict_mode:
                input_data(data=df[self.features])
            else:
                # 学習/評価: data, label, (任意で weight)
                kwargs = dict(data=df[self.features], label=df[self.target])
                if self.weight_col and self.weight_col in df.columns:
                    kwargs["weight"] = df[self.weight_col]
                input_data(**kwargs)

            del df
            return 1

    def _prepare_next_file(self):
        while self._current_file_index < len(self.paths):
            path = self.paths[self._current_file_index]
            self._current_file_index += 1
            dataset = ds.dataset(path, format="parquet")

            # fold フィルタ（predict_mode では原則無視）
            fexpr = None
            if (not self.predict_mode) and self.fold_col:
                col = ds.field(self.fold_col)
                if self.include_folds is not None:
                    fexpr = col.isin(sorted(self.include_folds))
                if self.exclude_folds is not None:
                    ex = ~col.isin(sorted(self.exclude_folds))
                    fexpr = ex if fexpr is None else (fexpr & ex)

            self._reader = dataset.scanner(
                columns=self._columns,
                batch_size=self.batch_rows,
                filter=fexpr,
            ).to_reader()
            return True
        self._reader = None
        return False


class XGBCVTrainer:
    """
    XGBを使ったCVトレーナー。

    Attributes
    ----------
    DATA_ID: int
        使用するdata
    params : dict
        XGBのパラメータ。
    early_stopping_rounds : int, default 100
        早期停止ラウンド数。
    num_boost_round : int, default 20000
        iterationの最大値。
    seed : int, default 42
        乱数シード。
    """

    def __init__(
        self,
        DATA_ID,
        base_dir,
        n_fold=5,
        params=None,
        early_stopping_rounds=200,
        num_boost_round=20000,
        seed=42
    ):
        self.data_id = DATA_ID
        self.base_dir = Path(base_dir)
        self.n_fold = n_fold
        self.params = params
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        self.fold_models = []
        self.fold_scores = []
        self.seed = seed
        self.oof_score = None

        self.default_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "learning_rate": 0.1,
            "max_depth": 7,
            "min_child_weight": 10.0,
            "gamma": 0,
            "colsample_bytree": 0.8,
            "subsample": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "verbosity": 0,
            "tree_method": "hist",
            "device": "cuda",
            "random_state": self.seed,
            "max_bin": 256,
            "grow_policy": "depthwise",
            "single_precision_histogram": True,
            "predictor": "gpu_predictor"
        }
        self.params = {**self.default_params, **(self.params or {})}

    def fit(
        self,
        features=None,
        target="target",
        cat_cols=None,
        fold_col=None,
        weight_col=None,
        batch_rows=1_000_000,
        use_cudf=True,
        extra_exclue_cols=None
    ):
        """
        CVを用いてモデルを学習し、OOF予測とtest_dfの平均予測を返す。

        Returns
        -------
        oof_preds : ndarray
            OOF予測配列
        test_preds : ndarray
            test_dfに対する予測配列
        """
        t_total_start = now()

        train_path = self.base_dir / f"tr_df{self.data_id}-seed{self.seed}.parquet"
        test_path = self.base_dir / f"test_df{self.data_id}.parquet"

        train_rows = pq.ParquetFile(train_path).metadata.num_rows
        test_rows = pq.ParquetFile(test_path).metadata.num_rows

        oof_preds = np.zeros(train_rows, dtype=np.float32)
        test_preds = np.zeros(test_rows,  dtype=np.float32)

        iteration_list = []
        fold_col = f"{self.n_fold}fold-seed{self.seed}"
        cat_cols = get_cat_cols(train_path)

        for i in range(self.n_fold):
            print("="*28)
            print(f"========== Fold {i + 1} ==========")
            print("="*28)
            t_fold_start = now()
            t_qdm_start = now()

            train_it = ParquetIter(
                paths=train_path,
                features=None,
                target="target",
                cat_cols=cat_cols,
                fold_col=fold_col,
                exclude_folds=[i],
                batch_rows=200000,
                use_cudf=True,
                keep_row_ids=True
            )
            valid_it = ParquetIter(
                paths=train_path,
                features=None,
                target="target",
                cat_cols=cat_cols,
                fold_col=fold_col,
                include_folds=[i],
                batch_rows=200000,
                use_cudf=True,
                keep_row_ids=True
            )
            test_it = ParquetIter(
                paths=test_path,
                features=None,
                target="target",
                cat_cols=cat_cols,
                fold_col=None,
                batch_rows=200000,
                predict_mode=True,
                use_cudf=True,
                keep_row_ids=False
            )

            dtrain = xgb.QuantileDMatrix(
                train_it, enable_categorical=True
            )
            dvalid = xgb.QuantileDMatrix(
                valid_it,
                enable_categorical=True, ref=dtrain
            )
            dtest = xgb.QuantileDMatrix(
                test_it,
                enable_categorical=True, ref=dtrain
            )

            val_idx = valid_it.collected_row_ids()

            evals_result = {}

            t_qdm_end = now()
            t_fit_start = now()

            model = xgb.train(
                self.params,
                dtrain,
                num_boost_round=self.num_boost_round,
                evals=[(dtrain, "train"), (dvalid, "eval")],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=100,
                evals_result=evals_result
            )

            # oof
            oof_preds[val_idx] = model.predict(
                dvalid, iteration_range=(0, model.best_iteration+1))
            test_preds += model.predict(
                dtest, iteration_range=(0, model.best_iteration+1))

            t_fit_end = now()
            t_fold_end = now()

            print_duration(
                t_qdm_start, t_qdm_end, "\nQuantileDMatrix Build Time"
            )
            print_duration(t_fit_start, t_fit_end)
            print_duration(t_fold_start, t_fold_end, f"Fold{i} Runtime")

            best_iter = model.best_iteration
            train_score = evals_result["train"]["auc"][best_iter]
            eval_score = evals_result["eval"]["auc"][best_iter]
            print(f"\nTrain AUC: {train_score:.5f}")
            print(f"Valid AUC: {eval_score:.5f}\n")

            """self.fold_models.append(
                XGBFoldModel(model, X_val, y_val, fold))"""
            self.fold_scores.append(eval_score)

            iteration_list.append(best_iter)

            del train_it, valid_it, test_it, dtrain, dvalid, dtest
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()

        print("="*32)
        print("========== CV Results ==========")
        print("="*32)
        print("Fold scores: [" + ", ".join(f"{x:.5f}" for x in self.fold_scores) + "]")
        print(
            f"Mean: {np.mean(self.fold_scores):.5f}, "
            f"Std: {np.std(self.fold_scores):.5f}"
        )
        dataset = ds.dataset(train_path, format="parquet")
        table = dataset.scanner(columns=[target]).to_table()
        y = table[target].combine_chunks().to_numpy().astype(np.float32, copy=False)

        self.oof_score = roc_auc_score(y, oof_preds)
        print(f"OOF score: {self.oof_score:.5f}")
        print(f"Avg best iteration: {np.mean(iteration_list)}")
        print(f"Best iterations: \n{iteration_list}")

        test_preds /= self.n_fold

        t_total_end = now()
        print_duration(t_total_start, t_total_end, "Total CV Runtime")

        return oof_preds, test_preds

    def full_train(self, iterations):
        """
        訓練データ全体でモデルを学習し、test_dfに対する予測結果をnpy形式で返す

        Parameters
        ----------
        iterations : int
            学習の繰り返し回数。

        Returns
        -------
        test_prads : np.ndarray
            test dataの予測値
        """
        t_total_start = now()

        train_path = self.base_dir / f"tr_df{self.data_id}-seed{self.seed}.parquet"
        test_path = self.base_dir / f"test_df{self.data_id}.parquet"

        test_rows = pq.ParquetFile(test_path).metadata.num_rows
        test_preds = np.zeros(test_rows,  dtype=np.float32)

        fold_col = f"{self.n_fold}fold-seed{self.seed}"
        cat_cols = get_cat_cols(train_path)

        t_qdm_start = now()

        train_it = ParquetIter(
            paths=train_path,
            features=None,
            target="target",
            cat_cols=cat_cols,
            fold_col=fold_col,
            batch_rows=200000,
            use_cudf=True,
            keep_row_ids=True
        )

        test_it = ParquetIter(
            paths=test_path,
            features=None,
            target="target",
            cat_cols=cat_cols,
            fold_col=None,
            batch_rows=200000,
            predict_mode=True,
            use_cudf=True,
            keep_row_ids=False
        )

        dtrain = xgb.QuantileDMatrix(
            train_it, enable_categorical=True
        )
        dtest = xgb.QuantileDMatrix(
            test_it,
            enable_categorical=True, ref=dtrain
        )
        t_qdm_end = now()
        t_fit_start = now()

        model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=int(iterations*1.25),
            evals=[]
        )
        t_fit_end = now()

        print_duration(
            t_qdm_start, t_qdm_end, "\nQuantileDMatrix Build Time"
        )
        print_duration(t_fit_start, t_fit_end)

        test_preds = model.predict(dtest)

        t_total_end = now()
        print_duration(t_total_start, t_total_end, "Total CV Runtime")

        return test_preds

    def fit_one_fold(self, fold=0):
        """
        指定した1つのfoldのみを用いてモデルを学習する。
        主にOptunaによるハイパーパラメータ探索時に使用。

        Parameters
        ----------
        fold : int
            学習に使うfold番号。

        Rerurn
        ------
        score : float
            Score
        """
        t_total_start = now()

        train_path = self.base_dir / f"tr_df{self.data_id}-seed{self.seed}.parquet"
        fold_col = f"{self.n_fold}fold-seed{self.seed}"
        cat_cols = get_cat_cols(train_path)

        evals_result = {}

        t_qdm_start = now()

        train_it = ParquetIter(
            paths=train_path,
            features=None,
            target="target",
            cat_cols=cat_cols,
            fold_col=fold_col,
            exclude_folds=[fold],
            batch_rows=200000,
            use_cudf=True,
            keep_row_ids=True
        )
        valid_it = ParquetIter(
            paths=train_path,
            features=None,
            target="target",
            cat_cols=cat_cols,
            fold_col=fold_col,
            include_folds=[fold],
            batch_rows=200000,
            use_cudf=True,
            keep_row_ids=True
        )

        dtrain = xgb.QuantileDMatrix(
            train_it, enable_categorical=True
        )
        dvalid = xgb.QuantileDMatrix(
            valid_it,
            enable_categorical=True, ref=dtrain
        )

        t_qdm_end = now()
        t_fit_start = now()

        model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=[(dtrain, "train"), (dvalid, "eval")],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=100,
            evals_result=evals_result,
            # callbacks=[wandb.xgboost.WandbCallback()]
        )

        t_fit_end = now()
        print_duration(
            t_qdm_start, t_qdm_end, "\nQuantileDMatrix Build Time"
        )
        print_duration(t_fit_start, t_fit_end)

        best_iter = model.best_iteration
        train_score = evals_result["train"]["auc"][best_iter]
        eval_score = evals_result["eval"]["auc"][best_iter]
        print(f"\nTrain AUC: {train_score:.5f}")
        print(f"Valid AUC: {eval_score:.5f}")

        del train_it, valid_it, dtrain, dvalid
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

        t_total_end = now()
        print_duration(t_total_start, t_total_end, "Total Runtime")

        return eval_score


class XGBFoldModel:
    """
    XGBのfold単位のモデルを保持するクラス。

    Attributes
    ----------
    model : xgb.Booster
        学習済みのXGBoostモデル。
    X_val : pd.DataFrame
        検証用の特徴量データ。
    y_val : pd.Series
        検証用のターゲットラベル。
    fold_index : int
        Foldの番号。
    """

    def __init__(self, model, X_val, y_val, fold_index):
        self.model = model
        self.X_valid = X_val
        self.y_valid = y_val
        self.fold_index = fold_index

    def shap_plot(self, sample=1000):
        """
        SHAPを用いた特徴量の重要度の可視化を行う。

        Parameters
        ----------
        sample : int, default 1000
            可視化に使用するサンプル数。
        """
        sample_X = self.X_valid[:sample].copy()
        for col in sample_X.select_dtypes(include="category").columns:
            sample_X[col] = sample_X[col].cat.codes

        explainer = shap.TreeExplainer(
            self.model, feature_perturbation='interventional')
        shap_values = explainer.shap_values(sample_X)
        shap.summary_plot(shap_values, sample_X)

    def plot_gain_importance(self):
        """
        特徴量のTotalGainに基づく重要度を棒グラフで可視化する。
        """
        importances = self.model.get_score(importance_type="total_gain")

        total_gain = sum(importances.values())
        importance_ratios = [
            np.round((v/total_gain)*100, 2)
            for k, v in importances.items()
        ]
        df = pd.DataFrame({
            "Feature": [k for k in importances.keys()],
            "ImportanceRatio": importance_ratios,
        }).sort_values("ImportanceRatio", ascending=False)

        fig, ax = plt.subplots(figsize=(12, max(4, len(df)*0.4)))
        sns.barplot(
            data=df,
            y="Feature",
            x="ImportanceRatio",
            orient="h",
            palette="viridis",
            hue="Feature",
            ax=ax
        )
        for container in ax.containers:
            labels = ax.bar_label(container)
            for label in labels:
                label.set_fontsize(20)
        plt.title("Feature Importance", fontsize=32)
        plt.xlabel("Importance", fontsize=28)
        plt.ylabel("Feature", fontsize=28)
        ax.tick_params(axis="x", labelsize=20)
        ax.tick_params(axis="y", labelsize=20)
        plt.tight_layout()
        plt.show()

    def save_model(self, path="../artifacts/model/xgb_vn.pkl"):
        """
        学習済みモデルを指定パスに保存する。

        Parameters
        ----------
        path : str
            モデルを保存するパス。
        """
        joblib.dump(self.model, path)

    def load_model(self, path):
        """
        指定されたパスからモデルを読み込む。

        Parameters
        ----------
        path : str
            モデルファイルのパス。

        Returns
        -------
        self : XGBFoldModel
            読み込んだモデルを保持するインスタンス自身を返す。
        """
        self.model = joblib.load(path)
        return self