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
from cuml.ensemble import RandomForestRegressor
from rmm.allocators.cupy import rmm_cupy_allocator
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score

from src.utils.loggers import CVLogger, NoOpLogger
from src.utils.print_duration import print_duration
from src.utils.mem_info import free_ram_gib, free_vram_gib


@dataclass
class RFRCVTrainer:
    """
    RFRを使ったGPUでのCVトレーナー。

    Attributes
    ----------
    tr_df : pd.DataFrame
        label付データ
    test_df : pd.DataFrame, default None
        labelなしデータ。CV学習はtest_df必須。
    params : dict
        RFRのパラメータ。
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

        self.rep_metric = "auc"
        self.metrics = {
            "rmse": lambda y, p: np.sqrt(mse(y, p)),
            "r2": r2_score,
            "mae": lambda y, p: np.mean(np.abs(y-p)),
            # "mape": lambda y, p: np.mean(np.abs((y-p)/y)),
            # "accuracy": accuracy_score,
            # "log_loss": log_loss,
            "auc": roc_auc_score
        }

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
            "n_folds": self.n_folds,
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

        fold_scores = {name: [] for name in self.metrics.keys()}

        test = cudf.read_parquet(self.test_paths, columns=self.features)

        for i in range(self.n_folds):
            title = f" Fold {i + 1} / {self.n_folds} "
            print("=" * 48)
            print(f"{title:=^48}")
            print("=" * 48)
            print(f"Free CPU Mem: {round(free_ram_gib(), 2)} GB")
            print(f"Free GPU Mem: {round(free_vram_gib(), 2)} GB")

            t_fold_start = now()
            fold_summary = {}

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

            model = RandomForestRegressor(**self.params)
            model.fit(X_train, y_train)

            pred = model.predict(X_valid).get()
            oof[val_idx] = pred
            test_pred += model.predict(test).get()

            for name, metric_func in self.metrics.items():
                val_score = metric_func(y_valid, pred)
                print(f"{name.upper()} Valid: {val_score:.5f}")
                fold_summary[name] = val_score
                fold_scores[name].append(val_score)

            t_fold_end = now()

            fold_summary["runtime"] = print_duration(
                t_fold_start, t_fold_end, f"Fold {i+1} Runtime"
            )

            for lg in loggers:
                lg.on_fold_end(
                    i,
                    "iter",
                    summary=fold_summary
                )

            del train, X_train, y_train, X_valid, y_valid
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

        oofs = {
            name: metric_func(y, oof)
            for name, metric_func in self.metrics.items()
        }

        oof_stats = {
            name: {
                "oof": oofs[name],
                "mean": np.mean(vals),
                "std": np.std(vals)
            }
            for name, vals in fold_scores.items()
        }

        print(f"\n{' CV Results ':*^48}")
        print("─" * 48)
        print(f" {'Metric':^9}  {'OOF':>10}  {'Mean':>10} ± {'Std':<10} ")
        print("-" * 48)
        for name, stats in oof_stats.items():
            print(f" {name.upper():^9} "
                  f" {stats['oof']:>10.5f} "
                  f" {stats['mean']:>10.5f} ± {stats['std']:<10.5f} ")
        print("─" * 48)

        print(f"Free CPU Mem: {round(free_ram_gib(), 2)} GB")
        print(f"Free GPU Mem: {round(free_vram_gib(), 2)} GB")

        result = {
            "oof": oof,
            "test_pred": test_pred,
            "oof_score": oofs[self.rep_metric]
        }
        overall_summary = {}
        for name, stats in oof_stats.items():
            overall_summary[f"{name}_mean"] = stats["mean"]
            overall_summary[f"{name}_std"] = stats["std"]
            overall_summary[f"{name}_oof"] = oofs[name]

        t_total_end = now()
        overall_summary["total_runtime"] = print_duration(
            t_total_start, t_total_end, "Total CV Runtime"
        )

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
        rmse : float
            Score
        """
        t_total_start = now()
        fold_summary = {}

        loggers = loggers or [NoOpLogger()]
        meta = {
            "data_id": self.data_id,
            "seed": self.seed,
            "n_folds": self.n_folds,
            **self.params
        }
        for lg in loggers:
            lg.on_start(meta)

        print(f"Free CPU Mem: {round(free_ram_gib(), 2)} GB")
        print(f"Free GPU Mem: {round(free_vram_gib(), 2)} GB")

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

        model = RandomForestRegressor(**self.params)
        model.fit(X_train, y_train)

        pred = model.predict(X_valid).get()

        for name, metric_func in self.metrics.items():
            val_score = metric_func(y_valid, pred)
            print(f"{name.upper()} Valid: {val_score:.5f}")
            fold_summary[name] = val_score

        del train, X_train, y_train, X_valid, y_valid
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        self.pmp.free_all_blocks()

        t_total_end = now()
        fold_summary["runtime"] = print_duration(
            t_total_start, t_total_end, "Total Runtime"
        )
        print(f"Free CPU Mem: {round(free_ram_gib(), 2)} GB")
        print(f"Free GPU Mem: {round(free_vram_gib(), 2)} GB")

        for lg in loggers:
            lg.on_fold_end(
                fold_idx,
                "iter",
                summary=fold_summary
            )

        return fold_summary[self.rep_metric]
