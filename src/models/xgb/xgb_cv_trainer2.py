import glob
import os
import time
from pathlib import Path

import cudf
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
from src.utils.make_fold_paths import make_fold_paths
from src.utils.print_duration import print_duration


class ParquetIter(xgb.core.DataIter):
    """
    ディスク上の Parquet をバッチで読み出し、XGBoost に逐次供給するための DataIter。
    - path は str でも list[str] でもOK
    - features: 学習に使う特徴量カラム名リスト
    - target: 目的変数カラム名
    - batch_rows: 1バッチの行数上限（大きすぎるとGPU/CPUメモリに乗らないことがある）
    - use_cudf: True なら cuDF データフレームで渡す（GPU学習時に有利）
    """

    def __init__(self, path, features, target, cat_cols=None, batch_rows=1_000_000, use_cudf=None):
        super().__init__()
        if isinstance(path, (str, os.PathLike)):
            path = [str(path)]
        self.paths = list(path)
        self.features = features
        self.cat_cols = cat_cols if cat_cols else []
        self.target = target
        self.batch_rows = int(batch_rows)
        self.use_cudf = use_cudf

        # 内部状態
        self._reader = None  # deque of RecordBatch
        self._current_file_index = 0  # どのファイルを読んでいるか

    # --- 必須: 反復の最初に呼ばれる ---
    def reset(self):
        self._current_file_index = 0
        self._reader = None

    # --- 必須: 次のバッチを input_data に詰めて 1 を返す。終端で 0 を返す ---
    def next(self, input_data):
        while True:
            if self._reader is None:
                if not self._prepare_next_file():
                    return 0
            try:
                batch = next(self._reader)
            except StopIteration:
                self._reader = None
                continue  # 次ファイルへ
            # Arrow -> pandas/cuDF
            df = cudf.DataFrame.from_arrow(batch) if self.use_cudf else batch.to_pandas()
            # カテゴリを数値化
            if self.cat_cols:
                df[self.cat_cols] = df[self.cat_cols].astype("category")
            input_data(data=df[self.features], label=df[self.target])
            del df  # 参照を切る（オプション）
            return 1

    # --- 内部: 次のファイルのバッチ列を準備する。準備できれば True ---
    def _prepare_next_file(self):
        while self._current_file_index < len(self.paths):
            path = self.paths[self._current_file_index]
            self._current_file_index += 1

            # 単一ファイルの dataset を作る（列を絞る）
            dataset = ds.dataset(path, format="parquet")
            cols = list(dict.fromkeys(self.features + [self.target]))

            self._reader = dataset.scanner(
                batch_size=self.batch_rows, columns=cols
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
        self.base_dir,
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
            "max_bin": 512,
            "grow_policy": "depthwise",
            "single_precision_histogram": True,
            "predictor": "gpu_predictor"
        }
        self.params = {**self.default_params, **(self.params or {})}

    def fit(self):
        """
        CVを用いてモデルを学習し、OOF予測とtest_dfの平均予測を返す。

        Returns
        -------
        oof_preds : ndarray
            OOF予測配列
        test_preds : ndarray
            test_dfに対する予測配列
        """
        oof_preds = np.zeros(len(self.X))
        test_preds = np.zeros(self.test.num_row())

        iteration_list = []
        fold_col = f"{self.n_fold}fold-seed{self.SEED}"

        for i in range(self.n_fold):
            print(f"\nFold {i + 1}")
            start = time.time()

            train_files, valid_file = make_fold_paths(
                self.base_dir,
                self.DATA_ID,
                self.seed,
                valid_fold_idx=i
            )
            cat_cols = get_cat_cols(train_files[0])

            train_it = ParquetIter(
                path=train_files,
                features=None,
                target="target",
                cat_cols=cat_cols,
                batch_rows=200000,
                use_cudf=True
            )

            valid_it = ParquetIter(
                path=valid_file,
                features=None,
                target="target",
                cat_cols=cat_cols,
                batch_rows=200000,
                use_cudf=True
            )

            dtrain = xgb.DMatrix(
                train_it, enable_categorical=True
            )
            dvalid = xgb.DMatrix(
                valid_it,
                enable_categorical=True
            )
            evals_result = {}

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
            oof_preds[val_idx] = model.predict(dvalid, iteration_range=(0, model.best_iteration+1))
            test_preds += model.predict(self.test, iteration_range=(0, model.best_iteration+1))

            end = time.time()
            print_duration(start, end)

            best_iter = model.best_iteration
            train_score = evals_result["train"]["auc"][best_iter]
            eval_score = evals_result["eval"]["auc"][best_iter]
            print(f"Train AUC: {train_score:.5f}")
            print(f"Valid AUC: {eval_score:.5f}")

            self.fold_models.append(
                XGBFoldModel(model, X_val, y_val, fold))
            self.fold_scores.append(eval_score)

            iteration_list.append(best_iter)

        print("\n=== CV Results ===")
        print(f"Fold scores: {self.fold_scores}")
        print(
            f"Mean: {np.mean(self.fold_scores):.5f}, "
            f"Std: {np.std(self.fold_scores):.5f}"
        )

        self.oof_score = roc_auc_score(self.y, oof_preds)
        print(f"OOF score: {self.oof_score:.5f}")
        print(f"Avg best iteration: {np.mean(iteration_list)}")
        print(f"Best iterations: \n{iteration_list}")

        test_preds /= self.n_splits

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
        if self.test is None:
            raise ValueError("test_df not provided for XGBCVTrainer.")

        dtrain = xgb.DMatrix(
            self.X, label=self.y,
            weight=self.weights, enable_categorical=True
        )

        start = time.time()

        model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=int(iterations*1.25),
            evals=[]
        )

        end = time.time()
        print_duration(start, end)

        test_preds = model.predict(self.test)

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
        tr_idx, val_idx = self.fold_indices[fold]
        start = time.time()

        X_tr, y_tr, w_tr = self.X.iloc[tr_idx], self.y[tr_idx], self.weights[tr_idx]
        X_val, y_val = self.X.iloc[val_idx], self.y[val_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr,
                             weight=w_tr, enable_categorical=True)
        dvalid = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

        evals_result = {}

        model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=[(dtrain, "train"), (dvalid, "eval")],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=100,
            evals_result=evals_result,
            callbacks=[wandb.xgboost.WandbCallback()]
        )

        end = time.time()
        print_duration(start, end)

        best_iter = model.best_iteration
        train_score = evals_result["train"]["auc"][best_iter]
        eval_score = evals_result["eval"]["auc"][best_iter]
        print(f"Train AUC: {train_score:.5f}")
        print(f"Valid AUC: {eval_score:.5f}")

        self.fold_models.append(
            XGBFoldModel(model, X_val, y_val, fold))

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