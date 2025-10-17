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
from cuml.linear_model import Ridge
from rmm.allocators.cupy import rmm_cupy_allocator
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss

from src.utils.loggers import CVLogger, NoOpLogger
from src.utils.print_duration import print_duration
from src.utils.mem_info import free_ram_gib, free_vram_gib


def compute_feature_stats(
    paths: list[str],
    features: list[str],
    num_cols: list[str],
    fold_col: str = None
):
    lf = pl.scan_parquet(paths, low_memory=True)

    exprs = []
    for c in num_cols:
        exprs += [pl.col(c).cast(pl.Float32).mean().alias(f"{c}_mean"),
                  pl.col(c).cast(pl.Float32).std(ddof=0).alias(f"{c}_std")]
    out = lf.select(exprs).collect(engine="streaming")
    mean = out.select(
        [f"{c}_mean" for c in num_cols]).to_numpy().ravel().astype(np.float32)
    std = out.select(
        [f"{c}_std" for c in num_cols]).to_numpy().ravel().astype(np.float32)
    std[std == 0] = 1.0
    return cp.asarray(mean, dtype=cp.float32), cp.asarray(std, dtype=cp.float32)


@dataclass
class RidgeCVTrainer:
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
            "alpha": 1.0,
            "fit_intercept": True,
            "solver": "auto"
        }
        self.params = {**default_params, **self.params}

        self.rep_metric = "auc"
        self.metrics = {
            "rmse": lambda y, p: np.sqrt(mse(y, p)),
            "r2": r2_score,
            "mae": lambda y, p: np.mean(np.abs(y-p)),
            # "mape": lambda y, p: np.mean(np.abs((y-p)/y)),
            "accuracy": lambda y, p: np.mean(
                [y_i == (1 if p_i > 0.5 else 0) for y_i, p_i in zip(y, p)]
            ),
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

        # Cat Colsを除外
        if self.features is None:
            meta = {
                c
                for c in ("row_id", self.target, self.weight_col, self.fold_col)
                if c and c in all_cols
            }
            if self.cat_cols:
                meta |= self.cat_cols
            pat = re.compile(r"^\d+fold(?:-[A-Za-z0-9]+)?$")
            self.features = [
                c for c in all_cols
                if c not in meta and not pat.fullmatch(c)
            ]

        self.mean, self.std = compute_feature_stats(
            self.train_paths,
            self.features,
            self.features,
        )

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

        test = cudf.read_parquet(self.test_paths, columns=self.features).to_cupy()

        fold_df = (
            pl.read_parquet(
                self.train_paths,
                columns=["row_id", self.fold_col]
            )
        )

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

            train = cudf.read_parquet(
                self.train_paths,
                columns=self.features + [self.target, self.fold_col]
            )

            X_train = (
                train[train[self.fold_col] != i]
                [self.features].to_cupy().astype(cp.float32)
            )
            y_train = (
                train[train[self.fold_col] != i]
                [self.target].to_cupy().astype(cp.float32)
            )

            X_valid = (
                train[train[self.fold_col] == i]
                [self.features].to_cupy().astype(cp.float32)
            )
            y_valid = (
                train[train[self.fold_col] == i]
                [self.target].to_cupy().get().astype(cp.float32)
            )

            X_train -= self.mean
            X_train /= (self.std + 1e-8)

            X_valid -= self.mean
            X_valid /= (self.std + 1e-8)

            val_idx = (
                fold_df
                .filter(pl.col(self.fold_col) == i)
                .get_column("row_id")
                .to_numpy()
                .astype(np.int32, copy=False)
            )

            model = Ridge(**self.params)
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
        t_total_start = now()
        fold_summary = {}

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
            train[train[self.fold_col] == fold_idx]
            [self.target].to_cupy().get()
        )

        X_train -= self.mean
        X_train /= (self.std + 1e-8)

        X_valid -= self.mean
        X_valid /= (self.std + 1e-8)

        model = Ridge(**self.params)
        model.fit(X_train, y_train)

        pred = model.predict(X_valid).get()

        for name, metric_func in self.metrics.items():
            val_score = metric_func(y_valid, pred)
            print(f"{name.upper()} Valid: {val_score:.5f}")
            fold_summary[name] = val_score

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

        del train, X_train, y_train, X_valid, y_valid
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        self.pmp.free_all_blocks()

        return fold_summary[self.rep_metric]