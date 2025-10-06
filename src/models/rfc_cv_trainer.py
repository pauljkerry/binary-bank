import gc
import os
import re
from dataclasses import dataclass, field
from time import perf_counter as now
from typing import Optional

import cudf
import rmm
import cupy as cp
import numpy as np
import polars as pl
import rmm.mr as mr
from cuml.ensemble import RandomForestClassifier
from rmm.allocators.cupy import rmm_cupy_allocator
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

from src.utils.loggers import CVLogger, NoOpLogger
from src.utils.print_duration import print_duration
from src.utils.mem_info import free_ram_gib, free_vram_gib


@dataclass
class RFCCVTrainer:
    """
    RFCを使ったGPUでのCVトレーナー。

    Attributes
    ----------
    tr_df : pd.DataFrame
        label付データ
    test_df : pd.DataFrame, default None
        labelなしデータ。CV学習はtest_df必須。
    params : dict
        RFCのパラメータ。
    n_splits : int, default 5
        KFoldの分割数。
    seed : int, default 42
        乱数シード。
    """
    data_id: str
    train_paths: str | list[str]
    test_paths: str | list[str] | None = None

    features: Optional[list[str]] = None

    target: str = "target"
    fold_col: Optional[str] = None
    weight_col: Optional[str] = None
    cat_cols: Optional[list[str]] = None

    params: dict = field(default_factory=dict)

    n_folds: int = 5
    seed: int = 42
    gpu: bool = True

    opts: dict = field(init=True, default_factory=dict)

    def __post_init__(self):
        if isinstance(self.train_paths, (str, os.PathLike)):
            self.train_paths = [str(self.train_paths)]
        else:
            self.train_paths = [str(p) for p in self.train_paths]

        if self.test_paths:
            if isinstance(self.test_paths, (str, os.PathLike)):
                self.test_paths = [str(self.test_paths)]
            else:
                self.test_paths = [str(p) for p in self.test_paths]

        self.lf_train = pl.scan_parquet(self.train_paths)

        default_params = {
            "n_estimators": 100,
            "max_depth": 16,
            "bootstrap": True,
            "random_state": self.seed,
            "n_streams": 1
        }

        self.params = {**default_params, **self.params}

        hdr = pl.read_parquet(self.train_paths, n_rows=0)
        all_cols = hdr.columns

        if self.fold_col is None:
            self.fold_col = f"{self.n_folds}fold-s{self.seed}"

        if self.cat_cols is None:
            self.cat_cols = [
                c for c, dt in zip(hdr.columns, hdr.dtypes)
                if dt == pl.Categorical
            ]

        if self.fold_col not in all_cols:
            raise ValueError(f"fold_col not found in dataset: {self.fold_col}")
        else:
            print(f"Fold Col: {self.fold_col}")

        if self.features is None:
            meta = {
                c
                for c in ("row_id", self.target, self.weight_col, self.fold_col)
                if c and c in all_cols
            }
            pat = re.compile(r"^\d+fold(?:-[A-Za-z0-9]+)?$")
            self.features = [
                c for c in all_cols
                if c not in meta and not pat.fullmatch(c)
            ]

        dev_mr = mr.CudaAsyncMemoryResource()
        mr.set_current_device_resource(dev_mr)
        rmm.reinitialize(
            managed_memory=False,
            initial_pool_size=None,
        )
        cp.cuda.set_allocator(rmm_cupy_allocator)

        cp.get_default_memory_pool().set_limit(4 * 1024**3)
        self.pmp = cp.cuda.PinnedMemoryPool()
        cp.cuda.set_pinned_memory_allocator(self.pmp.malloc)

    def fit(
        self,
        loggers: list[CVLogger] | None = None
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
        if self.test_paths is None:
            raise ValueError("Please provide test_paths (got None).")

        t_total_start = now()

        loggers = loggers or [NoOpLogger()]
        meta = {
            "data_id": self.data_id,
            "seed": self.seed,
            "n_folds": self.n_folds
            **self.params
        }
        for lg in loggers:
            lg.on_start(meta)

        train_rows = (
            self.lf_train
            .select(pl.len())
            .collect()
            .item()
        )
        test_rows = (
            pl.scan_parquet(self.test_paths)
              .select(pl.len())
              .collect()
              .item()
        )

        oof = np.zeros(train_rows, dtype=np.float32)
        test_pred = np.zeros(test_rows, dtype=np.float32)

        logloss_scores = []
        auc_scores = []

        test = cudf.read_parquet(self.test_paths, columns=self.features)

        for i in range(self.n_folds):
            title = f" Fold {i + 1} / {self.n_folds} "
            print("=" * 48)
            print(f"{title:=^48}")
            print("=" * 48)
            print(f"Free CPU Mem: {round(free_ram_gib(), 2)} GB")
            print(f"Free GPU Mem: {round(free_vram_gib(), 2)} GB")

            t_fold_start = now()

            train = cudf.read_parquet(
                self.train_paths,
                columns=self.features + [self.target, self.fold_col]
            )

            X_train = train[train[self.fold_col] != i][self.features].to_cupy()
            y_train = train[train[self.fold_col] != i][self.target].to_cupy()

            X_valid = train[train[self.fold_col] == i][self.features].to_cupy()
            y_valid = (
                self.lf_train
                .filter(pl.col(self.fold_col) == i)
                .select(self.target)
                .collect(engine="streaming")
                .to_numpy()
                .ravel()
            )

            val_idx = (
                pl.scan_parquet(self.train_paths)
                  .select(["row_id", self.fold_col])
                  .filter(pl.col(self.fold_col) == i)
                  .select("row_id")
                  .collect()
                  .get_column("row_id")
                  .to_numpy()
                  .astype(np.int32, copy=False)
            )

            model = RandomForestClassifier(**self.params)
            model.fit(X_train, y_train)

            pred = model.predict(X_valid).get()
            oof[val_idx] = pred
            test_pred += model.predict(test).get()

            logloss_valid = log_loss(y_valid, pred)
            auc_valid = roc_auc_score(y_valid, pred)

            t_fold_end = now()

            runtime = print_duration(
                t_fold_start, t_fold_end, f"Fold {i+1} Runtime"
            )

            print(f"Logloss Valid: {logloss_valid:.5f}")
            print(f"AUC Valid: {auc_valid:.5f}\n")

            logloss_scores.append(logloss_valid)
            auc_scores.append(auc_valid)

            fold_summary = {
                "logloss": logloss_valid,
                "auc": auc_valid,
                "runtime": runtime
            }

            for lg in loggers:
                lg.on_fold_end(
                    i,
                    "iter",
                    summary=fold_summary
                )

            del X_train, y_train, X_valid, y_valid
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            self.pmp.free_all_blocks()

        y = (
            pl.read_parquet(self.train_paths, columns=self.target)
              .get_column(self.target)
              .cast(pl.Float32)
              .to_numpy()
        )
        test_pred /= self.n_folds

        logloss_oof = log_loss(y, oof)
        logloss_mean = np.mean(logloss_scores)
        logloss_std = np.mean(logloss_scores)

        auc_oof = roc_auc_score(y, oof)
        auc_mean = np.mean(auc_scores)
        auc_std = np.mean(auc_scores)

        print(f"\n{' CV Results ':*^48}")
        print("─" * 48)
        print(f" {'Metric':^9}  {'OOF':>10}  {'Mean':>10} ± {'Std':<10} ")
        print("-" * 48)
        print(f" {'Logloss':^9} "
              f" {logloss_oof:>10.5f} "
              f" {logloss_mean:>10.5f} ± {logloss_std:<10.5f} ")
        print(f" {'AUC':^9} "
              f" {auc_oof:>10.5f} "
              f" {auc_mean:>10.5f} ± {auc_std:<10.5f} ")
        print("─" * 48)

        t_total_end = now()
        total_runtime = print_duration(
            t_total_start, t_total_end, "Total CV Runtime"
        )
        print(f"Free CPU Mem: {round(free_ram_gib(), 2)} GB")
        print(f"Free GPU Mem: {round(free_vram_gib(), 2)} GB")

        result = {
            "oof": oof,
            "test_pred": test_pred,
            "oof_score": logloss_oof
        }
        overall_summary = {
            "logloss_oof": logloss_oof,
            "logloss_mean": logloss_mean,
            "logloss_std": logloss_std,
            "auc_oof": auc_oof,
            "auc_mean": auc_mean,
            "auc_std": auc_std,
            "total_runtime": total_runtime
        }

        for lg in loggers:
            lg.on_end(overall_summary)

        return result

    def fit_one_fold(
        self,
        fold_idx=0,
        loggers=None
    ):
        """
        指定した1つのfoldのみを用いてモデルを学習する。
        主にOptunaによるハイパーパラメータ探索時に使用。

        Parameters
        ----------
        fold : int
            学習に使うfold番号。

        Rerurn
        ------
        """
        t_total_start = now()
        print(f"Free CPU Mem: {round(free_ram_gib(), 2)} GB")
        print(f"Free GPU Mem: {round(free_vram_gib(), 2)} GB")

        loggers = loggers or [NoOpLogger()]
        meta = {
            "data_id": self.data_id,
            "seed": self.seed,
            "n_folds": self.n_folds,
            **self.params
        }
        for lg in loggers:
            lg.on_start(meta)

        train = cudf.read_parquet(
            self.train_paths,
            columns=self.features + [self.target, self.fold_col]
        )

        X_train = train[train[self.fold_col] != fold_idx][self.features].to_cupy()
        y_train = train[train[self.fold_col] != fold_idx][self.target].to_cupy()

        X_valid = train[train[self.fold_col] == fold_idx][self.features].to_cupy()
        y_valid = (
            self.lf_train
            .filter(pl.col(self.fold_col) == fold_idx)
            .select(self.target)
            .collect(engine="streaming")
            .to_numpy()
            .ravel()
        )

        model = RandomForestClassifier(**self.params)
        model.fit(X_train, y_train)

        pred = model.predict(X_valid).get()

        logloss_valid = log_loss(y_valid, pred)
        auc_valid = roc_auc_score(y_valid, pred)

        print(f"Logloss Valid: {logloss_valid:.5f}")
        print(f"AUC Valid: {auc_valid:.5f}\n")

        del train, X_train, y_train, X_valid, y_valid
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        self.pmp.free_all_blocks()

        t_total_end = now()
        total_runtime = print_duration(
            t_total_start, t_total_end, "Total Runtime"
        )
        print(f"Free CPU Mem: {round(free_ram_gib(), 2)} GB")
        print(f"Free GPU Mem: {round(free_vram_gib(), 2)} GB")

        fold_summary = {
            "logloss": logloss_valid,
            "auc": auc_valid,
            "runtime": total_runtime
        }

        for lg in loggers:
            lg.on_fold_end(
                fold_idx,
                "iter",
                summary=fold_summary
            )

        return logloss_valid
