from __future__ import annotations
import gc
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional
from pathlib import Path
from time import perf_counter as now

import cudf
import cupy as cp
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from src.utils.get_cat_cols import get_cat_cols
from src.utils.logging import CVResult, CVLogger, NoOpLogger
from src.utils.print_duration import print_duration

try:
    import cudf

    _HAS_CUDF = True
except Exception:
    _HAS_CUDF = False


@dataclass(eq=False)
class ParquetIter(xgb.core.DataIter):
    # === 引数（元 __init__ のシグネチャ） ===
    paths: list[str] | str | os.PathLike
    features: Optional[list[str]] = None
    target: str = "target"
    cat_cols: Optional[Iterable[str]] = None
    fold_col: Optional[str] = None
    include_folds: Optional[Iterable[int]] = None
    exclude_folds: Optional[Iterable[int]] = None
    weight_col: Optional[str] = None
    batch_rows: int = 1_000_000
    use_cudf: Optional[bool] = None
    extra_exclude_cols: Optional[Iterable[str]] = None
    predict_mode: bool = False
    keep_row_ids: bool = True

    # === 内部状態（initの引数にしないもの） ===
    _temporary_data: Any = field(init=False, default=None, repr=False)
    _row_id_chunks: list[Any] = field(init=False, default_factory=list, repr=False)
    _pass_count: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        # 親の DataIter 初期化はここで
        super().__init__()

        # paths を正規化（文字列/PathLike -> list[str]）
        if isinstance(self.paths, (str, os.PathLike)):
            self.paths = [str(self.paths)]
        else:
            self.paths = [str(p) for p in self.paths]

        # 型・既定値の正規化
        self.batch_rows = int(self.batch_rows)
        self.cat_cols = list(self.cat_cols or [])
        self.include_folds = (
            None
            if self.include_folds is None
            else set(self.include_folds)
        )
        self.exclude_folds = (
            None
            if self.exclude_folds is None
            else set(self.exclude_folds)
        )
        self.use_cudf = (
            _HAS_CUDF
            if self.use_cudf is None
            else bool(self.use_cudf)
        )
        self.predict_mode = bool(self.predict_mode)
        self.keep_row_ids = bool(self.keep_row_ids)

        # --- extra exclude colsを除外 ---
        schema = ds.dataset(self.paths, format="parquet").schema
        all_cols = [f.name for f in schema]

        if self.extra_exclude_cols:
            excl = (
                {self.extra_exclude_cols}
                if isinstance(self.extra_exclude_cols, str)
                else set(self.extra_exclude_cols)
            )
            self.features = [c for c in self.features if c not in excl]

        # 入力列（重複除去）
        cols = list(self.features)
        if (not self.predict_mode) and (self.target in all_cols):
            cols.append(self.target)
        if (not self.predict_mode) and (self.weight_col in all_cols):
            cols.append(self.weight_col)
        if (not self.predict_mode) and (self.fold_col in all_cols):
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

            if self.cat_cols:
                for c in self.cat_cols:
                    if c in df.columns:
                        df[c] = df[c].astype("category")

            if (
                self.keep_row_ids
                and self._pass_count == 1
                and "row_id" in df.columns
            ):
                self._row_id_chunks.append(df["row_id"].to_numpy())

            if self.predict_mode:
                input_data(data=df[self.features])
            else:
                # 学習/評価: data, label, (任意で weight)
                kwargs = dict(data=df[self.features], label=df[self.target])
                if self.weight_col and self.weight_col in df.columns:
                    kwargs["weight"] = df[self.weight_col]
                input_data(**kwargs)

            del batch, df
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


@dataclass
class XGBCVTrainer:
    """
    XGBを使ったCVトレーナー。

    Attributes
    ----------
    data_id: int
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
    data_id: int
    feature_dir: str

    features: Optional[list[str]] = None

    target: str = "target"
    fold_col: Optional[str] = None
    weight_col: Optional[str] = None
    cat_cols: Optional[list[str]] = None

    params: dict = field(default_factory=dict)

    n_fold: int = 5
    seed: int = 42
    batch_rows: int = 1_000_000
    use_cudf: bool = True

    _fi_fold_frames: list = field(init=False, default_factory=list, repr=False)

    def __post_init__(self):
        self.feature_dir = Path(self.feature_dir)

        default_params = {
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
            "seed": self.seed,
            "max_bin": 256,
            "grow_policy": "depthwise",
            "predictor": "gpu_predictor",
            "early_stopping_rounds": 200,
            "num_boost_round": 20000
        }
        self.params = {**default_params, **(self.params or {})}
        self.early_stopping_rounds = int(
            self.params.pop("early_stopping_rounds")
        )
        self.num_boost_round = int(
            self.params.pop("num_boost_round")
        )

        self.train_path = (
            self.feature_dir /
            f"tr_df{self.data_id}-s{self.seed}.parquet"
        )
        self.test_path = (
            self.feature_dir /
            f"test_df{self.data_id}.parquet"
        )

        schema = ds.dataset(self.train_path, format="parquet").schema
        all_cols = [f.name for f in schema]

        if self.fold_col is None:
            self.fold_col = f"{self.n_fold}fold-s{self.seed}"

        if self.cat_cols is None:
            self.cat_cols = get_cat_cols(self.train_path)

        if self.fold_col not in all_cols:
            raise ValueError(f"fold_col not found in dataset: {self.fold_col}")
        else:
            print(f"Fold Col: {self.fold_col}")

        if self.features is None:
            meta = {"row_id"}
            if self.target in all_cols:
                meta.add(self.target)
            if self.weight_col in all_cols:
                meta.add(self.weight_col)
            if self.fold_col:
                meta.add(self.fold_col)
            self.features = [c for c in all_cols if c not in meta]

    def fit(
        self,
        extra_exclue_cols=None,
        loggers: list[CVLogger] | None = None
    ):
        """
        CVを用いてモデルを学習し、OOF予測とtest_dfの平均予測を返す。

        Returns
        -------
        oof_pred : ndarray
            OOF予測配列
        test_pred : ndarray
            test_dfに対する予測配列
        """
        t_total_start = now()

        loggers = loggers or [NoOpLogger()]
        meta = {
            "data_id": self.data_id,
            "seed": self.seed,
            "n_fold": self.n_fold,
            **self.params
        }
        for lg in loggers:
            lg.on_start(meta)

        train_rows = pq.ParquetFile(self.train_path).metadata.num_rows
        test_rows = pq.ParquetFile(self.test_path).metadata.num_rows

        oof = np.zeros(train_rows, dtype=np.float32)
        test_pred = np.zeros(test_rows, dtype=np.float32)

        iteration_list = []
        fold_scores = []

        for i in range(self.n_fold):
            print("=" * 22)
            print(f"===== Fold {i + 1} / {self.n_fold} =====")
            print("=" * 22)

            t_fold_start = now()
            t_qdm_start = now()

            train_it = ParquetIter(
                paths=self.train_path,
                features=self.features,
                target=self.target,
                cat_cols=self.cat_cols,
                fold_col=self.fold_col,
                exclude_folds=[i],
                batch_rows=self.batch_rows,
                use_cudf=True,
                keep_row_ids=True,
            )
            valid_it = ParquetIter(
                paths=self.train_path,
                features=self.features,
                target=self.target,
                cat_cols=self.cat_cols,
                fold_col=self.fold_col,
                include_folds=[i],
                batch_rows=self.batch_rows,
                use_cudf=self.use_cudf,
                keep_row_ids=True,
            )
            test_it = ParquetIter(
                paths=self.test_path,
                features=self.features,
                target=self.target,
                cat_cols=self.cat_cols,
                fold_col=None,
                batch_rows=self.batch_rows,
                predict_mode=True,
                use_cudf=self.use_cudf,
                keep_row_ids=False,
            )

            dtrain = xgb.QuantileDMatrix(
                train_it,
                enable_categorical=True
            )
            dvalid = xgb.QuantileDMatrix(
                valid_it,
                enable_categorical=True,
                ref=dtrain
            )
            dtest = xgb.QuantileDMatrix(
                test_it,
                enable_categorical=True,
                ref=dtrain
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
                evals_result=evals_result,
            )

            # oof
            oof[val_idx] = model.predict(
                dvalid, iteration_range=(0, model.best_iteration + 1)
            )
            test_pred += model.predict(
                dtest,
                iteration_range=(0, model.best_iteration + 1)
            )

            t_fit_end = now()
            t_fold_end = now()

            print_duration(t_qdm_start, t_qdm_end, "\nQuantileDMatrix Build Time")
            print_duration(t_fit_start, t_fit_end)
            print_duration(t_fold_start, t_fold_end, f"Fold {i} Runtime")

            best_iteration = model.best_iteration
            train_score = evals_result["train"]["auc"][best_iteration]
            eval_score = evals_result["eval"]["auc"][best_iteration]

            print(f"\nTrain AUC: {train_score:.5f}")
            print(f"Valid AUC: {eval_score:.5f}\n")

            iteration_list.append(best_iteration)
            fold_scores.append(eval_score)

            importances = model.get_score(importance_type="total_gain")
            total_gain = float(sum(importances.values()))
            df = pl.DataFrame(
                {
                    "Feature": list(importances.keys()),
                    "ImportanceRatio": [
                        (v / total_gain) * 100.0 for v in importances.values()
                    ],
                }
            )

            self._fi_fold_frames.append(df)
            for lg in loggers:
                lg.on_fold_end(
                    i,
                    eval_score,
                    evals_result,
                    best_iteration
                )

            del train_it, valid_it, test_it, dtrain, dvalid, dtest, model
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()

        print("=" * 32)
        print("========== CV Results ==========")
        print("=" * 32)
        print(
            "Fold scores: [" + ", ".join(f"{x:.5f}" for x in fold_scores) + "]"
        )
        mean = np.mean(fold_scores)
        std = np.std(fold_scores)
        print(
            f"Mean: {mean:.5f}, "
            f"Std: {std:.5f}"
        )
        dataset = ds.dataset(self.train_path, format="parquet")
        table = dataset.scanner(columns=[self.target]).to_table()
        y = table[self.target].combine_chunks().to_numpy().astype(np.float32, copy=False)

        oof_score = roc_auc_score(y, oof)
        print(f"OOF score: {oof_score:.5f}")
        print(f"Avg best iteration: {np.mean(iteration_list)}")
        print(f"Best iterations: \n{iteration_list}")

        test_pred /= self.n_fold

        all_fi = pl.concat(self._fi_fold_frames, how="vertical_relaxed")
        fi_mean = (
            all_fi
            .group_by("Feature")
            .agg([
                (pl.sum("ImportanceRatio") / self.n_fold).alias("mean_ratio")
            ])
        ).sort("mean_ratio", descending=True)

        result = CVResult(
            oof,
            test_pred,
            oof_score,
            mean,
            std,
            iteration_list,
            fi_mean
        )
        for lg in loggers:
            lg.on_end(result)

        t_total_end = now()
        print_duration(t_total_start, t_total_end, "Total CV Runtime")

        return result

    def full_train(self):
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

        test_rows = pq.ParquetFile(self.test_path).metadata.num_rows
        test_pred = np.zeros(test_rows, dtype=np.float32)

        self.num_boost_round = (
            self.num_boost_round * (self.n_fold/(self.n_fold-1))
        )

        t_qdm_start = now()

        train_it = ParquetIter(
            paths=self.train_path,
            features=self.features,
            target=self.target,
            cat_cols=self.cat_cols,
            fold_col=self.fold_col,
            batch_rows=self.batch_rows,
            use_cudf=self.use_cudf,
            keep_row_ids=True,
        )

        test_it = ParquetIter(
            paths=self.test_path,
            features=self.features,
            target=self.target,
            cat_cols=self.cat_cols,
            fold_col=None,
            batch_rows=self.batch_rows,
            predict_mode=True,
            use_cudf=self.use_cudf,
            keep_row_ids=False,
        )

        dtrain = xgb.QuantileDMatrix(train_it, enable_categorical=True)
        dtest = xgb.QuantileDMatrix(test_it, enable_categorical=True, ref=dtrain)
        t_qdm_end = now()
        t_fit_start = now()

        model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=[]
        )
        t_fit_end = now()

        print_duration(t_qdm_start, t_qdm_end, "\nQuantileDMatrix Build Time")
        print_duration(t_fit_start, t_fit_end)

        test_pred = model.predict(dtest)

        t_total_end = now()
        print_duration(t_total_start, t_total_end, "Total CV Runtime")

        return test_pred

    def fit_one_fold(self, fold_idx=0, loggers=None):
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

        loggers = loggers or [NoOpLogger()]
        meta = {
            "data_id": self.data_id,
            "seed": self.seed,
            "n_fold": self.n_fold,
            **self.params
        }
        for lg in loggers:
            lg.on_start(meta)

        evals_result = {}

        t_qdm_start = now()

        train_it = ParquetIter(
            paths=self.train_path,
            features=self.features,
            target=self.target,
            cat_cols=self.cat_cols,
            fold_col=self.fold_col,
            exclude_folds=[fold_idx],
            batch_rows=self.batch_rows,
            use_cudf=self.use_cudf,
            keep_row_ids=True,
        )
        valid_it = ParquetIter(
            paths=self.train_path,
            features=self.features,
            target=self.target,
            cat_cols=self.cat_cols,
            fold_col=self.fold_col,
            include_folds=[fold_idx],
            batch_rows=self.batch_rows,
            use_cudf=self.use_cudf,
            keep_row_ids=True,
        )

        dtrain = xgb.QuantileDMatrix(
            train_it,
            enable_categorical=True
        )
        dvalid = xgb.QuantileDMatrix(
            valid_it,
            enable_categorical=True,
            ref=dtrain
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
        )

        t_fit_end = now()
        print_duration(t_qdm_start, t_qdm_end, "\nQuantileDMatrix Build Time")
        print_duration(t_fit_start, t_fit_end)

        best_iteration = model.best_iteration

        train_score = evals_result["train"]["auc"][best_iteration]
        eval_score = evals_result["eval"]["auc"][best_iteration]
        print(f"\nTrain AUC: {train_score:.5f}")
        print(f"Valid AUC: {eval_score:.5f}")

        importances = model.get_score(importance_type="total_gain")

        for lg in loggers:
            lg.on_fold_end(
                fold_idx,
                eval_score,
                evals_result,
                best_iteration,
                importances
            )

        del train_it, valid_it, dtrain, dvalid, model
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

        t_total_end = now()
        print_duration(t_total_start, t_total_end, "Total Runtime")

        return eval_score
