import gc
import os
import re
from dataclasses import dataclass, field
from time import perf_counter as now
from typing import Optional

import numpy as np
import polars as pl
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

from src.utils.loggers import CVLogger, NoOpLogger
from src.utils.print_duration import print_duration
from src.utils.mem_info import free_ram_gib, free_vram_gib


@dataclass(eq=False)
class CBCVTrainer:
    """
    CBを使ったCVトレーナー。

    Attributes
    ----------
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
        self.lf_test = (
            pl.scan_parquet(self.test_paths) if self.test_paths else None
        )

        default_params = {
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "learning_rate": 0.1,
            "depth": 6,
            "iterations": 20000,
            "min_data_in_leaf": 1,
            "l2_leaf_reg": 3.0,
            "bagging_temperature": 1,
            "random_strength": 10,
            "border_count": 128,
            "grow_policy": "SymmetricTree",
            "random_seed": self.seed,
            "task_type": "GPU",  # or CPU
            "early_stopping_rounds": 100,
            "allow_writing_files": False,
            "verbose": 100
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

    def fit(
        self,
        loggers: list[CVLogger] | None = None
    ):
        """
        CVを用いてモデルを学習し、OOF予測とtest_dfの平均予測を返す。

        Returns
        -------
        """
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
        auc_scores = []

        for i in range(self.n_folds):
            title = f" Fold {i + 1} / {self.n_folds} "
            print("=" * 48)
            print(f"{title:=^48}")
            print("=" * 48)
            print(f"Free CPU Mem: {round(free_ram_gib(), 2)} GB")
            print(f"Free GPU Mem: {round(free_vram_gib(), 2)} GB")

            t_fold_start = now()

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
                valid
                .select("row_id")
                .to_series()
                .to_numpy()
                .astype(np.int32, copy=False)
            )
            test = (
                self.lf_test
                .select(self.features)
                .collect(engine="streaming")
                .to_numpy()
                .astype(np.float32)
            )

            train_pool = Pool(
                X_train,
                y_train,
                cat_features=self.cat_cols
            )
            valid_pool = Pool(
                X_valid,
                y_valid,
                cat_features=self.cat_cols
            )

            model = CatBoostClassifier(**self.params)

            model.fit(
                train_pool,
                eval_set=valid_pool,
                use_best_model=True
            )

            best_iter = model.best_iteration_

            val_pred = model.predict_proba(X_valid, ntree_end=best_iter)[:, 1]
            oof[val_idx] = val_pred

            test_pred += model.predict_proba(test)[:, 1]

            valid_auc = roc_auc_score(y_valid, val_pred)
            print(f"AUC Valid: {valid_auc:.5f}")

            auc_scores.append(valid_auc)
            iteration_list.append(best_iter)

            t_fold_end = now()
            runtime = print_duration(
                t_fold_start, t_fold_end, f"Fold {i+1} Runtime"
            )
            fold_summary = {
                "auc": valid_auc,
                "runtime": runtime
            }

            for lg in loggers:
                lg.on_fold_end(
                    i,
                    axis_name="iter",
                    summary=fold_summary
                )
            del model, X_train, y_train, X_valid, y_valid, test
            gc.collect()

        y = (
            self.lf_train
            .select(self.target)
            .cast(pl.Float32)
            .collect(engine="streaming")
            .to_numpy()
            .ravel()
        )
        test_pred /= self.n_folds
        auc_oof = roc_auc_score(y, oof)

        auc_mean = np.mean(auc_scores)
        auc_std = np.std(auc_scores)
        iter_mean = np.mean(iteration_list)

        print(f"\n{' CV Results ':*^48}")
        print("─" * 48)
        print(f" {'Metric':^9}  {'OOF':>10}  {'Mean':>10} ± {'Std':<10} ")
        print("-" * 48)
        print(f" {'AUC':^9} "
              f" {auc_oof:>10.5f} "
              f" {auc_mean:>10.5f} ± {auc_std:<10.5f} ")
        print("─" * 48)

        print(f"Avg best iteration: {np.mean(iteration_list)}")

        t_total_end = now()
        total_runtime = print_duration(
            t_total_start, t_total_end, "Total CV Runtime"
        )
        print(f"Free CPU Mem: {round(free_ram_gib(), 2)} GB")
        print(f"Free GPU Mem: {round(free_vram_gib(), 2)} GB")

        result = {
            "oof": oof,
            "test_pred": test_pred,
            "oof_score": auc_oof
        }
        overall_summary = {
            "auc_oof": auc_oof,
            "auc_mean": auc_mean,
            "auc_std": auc_std,
            "iter_mean": iter_mean,
            "total_runtime": total_runtime
        }
        for lg in loggers:
            lg.on_end(overall_summary)

        return result

    def full_train(
        self,
        loggers: list[CVLogger] | None = None
    ):
        """
        訓練データ全体でモデルを学習し、test_dfに対する予測結果をnpy形式で保存する。

        Parameters
        ----------
        """
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

        train_pool = Pool(
            X_train,
            y_train,
            cat_features=self.cat_cols,
            weight=self.weights
        )

        model = CatBoostClassifier(**self.params)

        model.fit(
            train_pool,
            use_best_model=True
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
        fold_idx=0,
        loggers: list[CVLogger] | None = None
    ):
        """
        指定した1つのfoldのみを用いてモデルを学習する。
        主にOptunaによるハイパーパラメータ探索時に使用。

        Parameters
        ----------

        Return
        ------
        auc : float
            Score
        """
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

        train_pool = Pool(
            X_train,
            y_train,
            cat_features=self.cat_cols
        )
        valid_pool = Pool(
            X_valid,
            y_valid,
            cat_features=self.cat_cols
        )

        model = CatBoostClassifier(**self.params)

        model.fit(
            train_pool, eval_set=valid_pool, use_best_model=True,
        )

        val_pred = model.predict_proba(X_valid)[:, 1]
        auc_valid = roc_auc_score(y_valid, val_pred)
        print(f"Valid AUC: {auc_valid:.5f}")

        del model, X_train, y_train, X_valid, y_valid
        gc.collect()

        t_total_end = now()
        total_runtime = print_duration(
            t_total_start, t_total_end, "Total CV Runtime"
        )
        print(f"Free CPU Mem: {round(free_ram_gib(), 2)} GB")
        print(f"Free GPU Mem: {round(free_vram_gib(), 2)} GB")

        fold_summary = {
            "auc": auc_valid,
            "runtime": total_runtime
        }

        for lg in loggers:
            lg.on_fold_end(
                fold_idx,
                axis_name="iter",
                summary=fold_summary
            )

        return auc_valid
