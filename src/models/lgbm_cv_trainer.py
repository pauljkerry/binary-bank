import gc
import math
import os
import re
from dataclasses import dataclass, field
from time import perf_counter as now
from typing import Optional

import lightgbm as lgb
import numpy as np
import polars as pl

from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

from src.utils.loggers import CVLogger, NoOpLogger
from src.utils.print_duration import print_duration
from src.utils.mem_info import free_ram_gib, free_vram_gib


@dataclass(eq=False)
class LGBMCVTrainer:
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
        self.lf_test = (
            pl.scan_parquet(self.test_paths) if self.test_paths else None
        )

        default_params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.1,
            "num_leaves": 500,
            "max_depth": -1,
            "min_child_samples": 100,
            "min_split_gain": 0,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "lambda_l1": 0.0,
            "lambda_l2": 0.0,
            "n_jobs": 25,
            "verbosity": -1,
            "random_state": self.seed
        }

        user_params = self.params or {}
        merged = {**default_params, **user_params}

        self.early_stopping_rounds = self.opts.get(
            "early_stopping_rounds",
            None
        )
        self.num_boost_round = self.opts.get(
            "num_boost_round",
            20000
        )
        if self.early_stopping_rounds is None:
            lr = float(merged["learning_rate"])
            self.early_stopping_rounds = max(50, int(math.ceil(10.0 / lr)))

        self.params = merged

        self.rep_metric = "auc"
        self.metrics = {
            "rmse": lambda y, p: np.sqrt(mse(y, p)),
            "r2": r2_score,
            "mae": lambda y, p: np.mean(np.abs(y-p)),
            # "mape": lambda y, p: np.mean(np.abs((y-p)/y)),
            "accuracy": lambda y, p: np.mean(
                [y_i == (1 if p_i > 0.5 else 0) for y_i, p_i in zip(y, p)]
            ),
            "log_loss": log_loss,
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
            "early_stopping_rounds": self.early_stopping_rounds,
            **self.params
        }
        for lg in loggers:
            lg.on_start(meta)

        train_rows = (
            pl.scan_parquet(self.train_paths)
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

        iteration_list = []
        fold_scores = {name: [] for name in self.metrics.keys()}
        fi_fold_frames = []

        test = (
            self.lf_test
            .select(self.features)
            .collect(engine="streaming")
            .to_numpy()
            .astype(np.float32)
        )

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

            need_cols = self.features + [self.target, "row_id"]
            train = (
                self.lf_train
                .filter(pl.col(self.fold_col) != i)
                .select(need_cols)
                .collect(engine="streaming")
            )
            valid = (
                self.lf_train
                .filter(pl.col(self.fold_col) == i)
                .select(need_cols)
                .collect(engine="streaming")
            )
            X_train = (
                train
                .select(self.features)
                .to_numpy()
                .astype(np.float32)
            )
            y_train = (
                train
                .select(self.target)
                .to_numpy()
                .astype(np.int32)
                .ravel()
            )
            X_valid = (
                valid
                .select(self.features)
                .to_numpy()
                .astype(np.float32)
            )
            y_valid = (
                valid
                .select(self.target)
                .to_numpy()
                .astype(np.int32)
                .ravel()
            )
            val_idx = (
                fold_df
                .filter(pl.col(self.fold_col) == i)
                .get_column("row_id")
                .to_numpy()
                .astype(np.int32, copy=False)
            )

            dtrain = lgb.Dataset(
                X_train,
                label=y_train,
                feature_name=self.features,
                categorical_feature=self.cat_cols,
            )

            dvalid = lgb.Dataset(
                X_valid,
                label=y_valid,
                feature_name=self.features,
                reference=dtrain
            )

            evals_result = {}

            model = lgb.train(
                self.params,
                dtrain,
                num_boost_round=self.num_boost_round,
                valid_sets=[dtrain, dvalid],
                valid_names=["train", "eval"],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=self.early_stopping_rounds),
                    lgb.record_evaluation(evals_result),
                    lgb.log_evaluation(period=100)
                ]
            )

            val_pred = model.predict(X_valid)
            oof[val_idx] = val_pred
            test_pred += model.predict(test)

            best_iter = model.best_iteration
            fold_summary["best_iter"] = best_iter

            for name, metric_func in self.metrics.items():
                val_score = metric_func(y_valid, val_pred)
                print(f"{name.upper()} Valid: {val_score:.5f}")
                fold_summary[name] = val_score
                fold_scores[name].append(val_score)

            iteration_list.append(best_iter)

            importances = model.feature_importance(importance_type="gain")
            total_gain = importances.sum()
            df = pl.DataFrame(
                {
                    "Feature": model.feature_name(),
                    "Importance": [
                        ((v/total_gain)*100.0)/self.n_folds for v in importances
                    ],
                }
            )
            fi_fold_frames.append(df)

            t_fold_end = now()
            fold_summary["runtime"] = print_duration(
                t_fold_start, t_fold_end, f"\nFold {i+1} Runtime"
            )

            for lg in loggers:
                lg.on_fold_end(
                    i,
                    axis_name="iter",
                    evals_result=evals_result,
                    summary=fold_summary
                )
            del model, X_train, y_train, X_valid, y_valid
            gc.collect()

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

        all_fi = pl.concat(fi_fold_frames, how="vertical_relaxed")
        fi_mean = (
            all_fi
            .group_by("Feature")
            .agg([
                pl.sum("Importance").alias("Importance")
            ])
        ).sort("Importance", descending=True)

        iter_mean = np.mean(iteration_list)

        print(f"\n{' CV Results ':*^48}")
        print("─" * 48)
        print(f" {'Metric':^9}  {'OOF':>10}  {'Mean':>10} ± {'Std':<10} ")
        print("-" * 48)
        for name, stats in oof_stats.items():
            print(f" {name.upper():^9} "
                  f" {stats['oof']:>10.5f} "
                  f" {stats['mean']:>10.5f} ± {stats['std']:<10.5f} ")
        print("─" * 48)
        print(f"Avg best iteration: {iter_mean}")

        print(f"Free CPU Mem: {round(free_ram_gib(), 2)} GB")
        print(f"Free GPU Mem: {round(free_vram_gib(), 2)} GB")

        result = {
            "oof": oof,
            "test_pred": test_pred,
            "oof_score": oofs[self.rep_metric],
            "fi_mean": fi_mean
        }
        overall_summary = {"iter_mean": iter_mean}
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

    def full_train(
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
            "early_stopping_rounds": self.early_stopping_rounds,
            **self.params
        }
        for lg in loggers:
            lg.on_start(meta)

        test_rows = (
            pl.scan_parquet(self.test_paths)
              .select(pl.len())
              .collect()
              .item()
        )

        test_pred = np.zeros(test_rows, dtype=np.float32)

        need_cols = self.features + [self.target]
        train = (
            self.lf_train
            .select(need_cols)
            .collect(engine="streaming")
        )
        X_train = (
            train
            .select(self.features)
            .to_numpy()
            .astype(np.float32)
        )
        y_train = (
            train
            .select(self.target)
            .to_numpy()
            .astype(np.int32)
            .ravel()
        )
        test = (
            self.lf_test
            .select(self.features)
            .collect(engine="streaming")
            .to_numpy()
            .astype(np.float32)
        )

        dtrain = lgb.Dataset(
            X_train,
            label=y_train,
            feature_name=self.features,
            categorical_feature=self.cat_cols,
        )

        model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            valid_sets=[dtrain],
            valid_names=["train"]
        )

        test_pred += model.predict(test)

        test_pred /= self.n_folds

        t_total_end = now()
        total_runtime = print_duration(
            t_total_start, t_total_end, "Total Runtime"
        )
        print(f"Free CPU Mem: {round(free_ram_gib(), 2)} GB")
        print(f"Free GPU Mem: {round(free_vram_gib(), 2)} GB")

        result = {
            "oof": None,
            "test_pred": test_pred,
            "oof_score": None
        }
        overall_summary = {
            "total_runtime": total_runtime
        }
        for lg in loggers:
            lg.on_end(overall_summary)

        return result

    def fit_one_fold(
        self,
        fold_idx: int = 0,
        loggers: list[CVLogger] | None = None
    ) -> float:
        t_total_start = now()
        fold_summary = {}

        loggers = loggers or [NoOpLogger()]
        meta = {
            "data_id": self.data_id,
            "seed": self.seed,
            "n_folds": self.n_folds,
            "early_stopping_rounds": self.early_stopping_rounds,
            **self.params
        }
        for lg in loggers:
            lg.on_start(meta)

        print(f"Free CPU Mem: {round(free_ram_gib(), 2)} GB")
        print(f"Free GPU Mem: {round(free_vram_gib(), 2)} GB")

        need_cols = self.features + [self.target, "row_id"]
        train = (
            self.lf_train
            .filter(pl.col(self.fold_col) != fold_idx)
            .select(need_cols)
            .collect(engine="streaming")
        )
        valid = (
            self.lf_train
            .filter(pl.col(self.fold_col) == fold_idx)
            .select(need_cols)
            .collect(engine="streaming")
        )
        X_train = (
            train
            .select(self.features)
            .to_numpy()
            .astype(np.float32)
        )
        y_train = (
            train
            .select(self.target)
            .to_numpy()
            .astype(np.int32)
            .ravel()
        )
        X_valid = (
            valid
            .select(self.features)
            .to_numpy()
            .astype(np.float32)
        )
        y_valid = (
            valid
            .select(self.target)
            .to_numpy()
            .astype(np.int32)
            .ravel()
        )

        dtrain = lgb.Dataset(
            X_train,
            label=y_train,
            feature_name=self.features,
            categorical_feature=self.cat_cols,
        )

        dvalid = lgb.Dataset(
            X_valid,
            label=y_valid,
            feature_name=self.features,
            reference=dtrain
        )

        evals_result = {}

        model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.early_stopping_rounds),
                lgb.record_evaluation(evals_result),
                lgb.log_evaluation(period=100)
            ]
        )

        fold_summary["best_iter"] = model.best_iteration

        val_pred = model.predict(X_valid)
        for name, metric_func in self.metrics.items():
            val_score = metric_func(y_valid, val_pred)
            print(f"{name.upper()} Valid: {val_score:.5f}")
            fold_summary[name] = val_score

        del model, X_train, y_train, X_valid, y_valid
        gc.collect()

        t_total_end = now()
        fold_summary["runtime"] = print_duration(
            t_total_start, t_total_end, "Total CV Runtime"
        )
        print(f"Free CPU Mem: {round(free_ram_gib(), 2)} GB")
        print(f"Free GPU Mem: {round(free_vram_gib(), 2)} GB")

        for lg in loggers:
            lg.on_fold_end(
                fold_idx,
                axis_name="iter",
                evals_result=evals_result,
                summary=fold_summary
            )

        return fold_summary[self.rep_metric]
