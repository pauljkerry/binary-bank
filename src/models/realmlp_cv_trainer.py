import gc
import os
import re
from dataclasses import dataclass, field
from time import perf_counter as now
from typing import Optional

import rmm
import cupy as cp
import numpy as np
import polars as pl
import rmm.mr as mr
from pytabkit import RealMLP_TD_Classifier
from rmm.allocators.cupy import rmm_cupy_allocator
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score

from src.utils.loggers import CVLogger, NoOpLogger
from src.utils.print_duration import print_duration
from src.utils.mem_info import free_ram_gib, free_vram_gib


@dataclass
class RealMLPCVTrainer:
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
            'device': 'cuda',
            'n_epochs': 10,
            'random_state': 42,
            'verbosity': 2,
            'hidden_sizes': [64, 64, 64, 64, 64],
            'max_one_hot_cat_size': 9,
            'embedding_size': 8,
            'weight_param': 'ntk',
            'weight_init_mode': 'std',
            'bias_init_mode': 'he+5',
            'bias_lr_factor': 0.1,
            'act': 'mish',
            'use_parametric_act': True,
            'act_lr_factor': 0.1,
            'wd': 0.0,
            'wd_sched': 'flat_cos',
            'bias_wd_factor': 0.0,
            'block_str': 'w-b-a-d',
            'p_drop': 0.15,
            'p_drop_sched': 'flat_cos',
            'add_front_scale': False,
            'scale_lr_factor': 6.0,
            'tfms': [
                'one_hot',
                'median_center',
                'robust_scale',
                'smooth_clip',
                'embedding'
            ],
            'num_emb_type': 'pbld',
            'plr_sigma': 0.28992671701332556,
            'plr_hidden_1': 16,
            'plr_hidden_2': 4,
            'plr_lr_factor': 0.1,
            'clamp_output': True,
            'normalize_output': True,
            'lr': 0.1400853680319456,
            'lr_sched': 'coslog4',
            'opt': 'adam',
            'sq_mom': 0.95,
        }

        self.params = {**default_params, **self.params}

        self.rep_metric = "auc"
        self.metrics = {
            "accuracy": accuracy_score,
            "log_loss": log_loss,
            "auc": roc_auc_score
        }

        hdr = pl.read_parquet(self.train_paths, n_rows=0)
        all_cols = hdr.columns

        if self.fold_col is None:
            self.fold_col = f"{self.n_folds}fold-s{self.seed}"

        if self.cat_cols is None:
            self.cat_cols = [
                c for c, dt in hdr.schema.items()
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
    ) -> dict:
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

        test = pl.read_parquet(self.test_paths, columns=self.features)

        for i in range(self.n_folds):
            title = f" Fold {i + 1} / {self.n_folds} "
            print("=" * 48)
            print(f"{title:=^48}")
            print("=" * 48)
            print(f"Free CPU Mem: {round(free_ram_gib(), 2)} GB")
            print(f"Free GPU Mem: {round(free_vram_gib(), 2)} GB")

            t_fold_start = now()
            fold_summary = {}

            train = pl.read_parquet(
                self.train_paths,
                columns=self.features + [self.target, self.fold_col]
            )

            X_train = train[train[self.fold_col] != i][self.features].to_numpy()
            y_train = train[train[self.fold_col] != i][self.target].to_numpy()

            X_valid = train[train[self.fold_col] == i][self.features].to_numpy()
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

            model = RealMLP_TD_Classifier(**self.params)
            model.fit(X_train, y_train)

            pred = model.predict(X_valid)
            oof[val_idx] = pred
            test_pred += model.predict(test)

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
        fold_idx: int = 0,
        loggers: list[CVLogger] | None = None
    ) -> float:
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

        train = pl.read_parquet(
            self.train_paths,
            columns=self.features + [self.target, self.fold_col]
        )

        X_train = (
            train.filter(pl.col(self.fold_col) != fold_idx)
            .select(self.features)
            .to_numpy()
        )
        y_train = (
            train.filter(pl.col(self.fold_col) != fold_idx)
            .select(self.target)
            .to_numpy()
            .ravel()
        )

        X_valid = (
            train.filter(pl.col(self.fold_col) == fold_idx)
            .select(self.features)
            .to_numpy()
        )
        y_valid = (
            self.lf_train
            .filter(pl.col(self.fold_col) == fold_idx)
            .select(self.target)
            .collect(engine="streaming")
            .to_numpy()
            .ravel()
        )

        model = RealMLP_TD_Classifier(**self.params)
        model.fit(X_train, y_train, X_valid, y_valid, cat_col_names=self.cat_cols)

        pred = model.predict(X_valid)

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
