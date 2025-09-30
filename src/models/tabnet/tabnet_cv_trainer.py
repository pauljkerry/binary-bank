import gc
import os
from dataclasses import dataclass, field
from typing import Iterable, Optional
from time import perf_counter as now

import torch
import numpy as np
import polars as pl

from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

from src.utils.loggers import CVResult, CVLogger, NoOpLogger
from src.utils.print_duration import print_duration
from src.utils.mem_info import free_ram_gib, free_vram_gib


def compute_feature_stats(
    paths: list[str],
    features: list[str],
    num_cols: list[str],
    fold_col: str = None,
    include_folds: Optional[Iterable[int]] = None,
    exclude_folds: Optional[Iterable[int]] = None
):
    lf = pl.scan_parquet(paths, low_memory=True)
    if fold_col:
        if include_folds is not None:
            lf = lf.filter(pl.col(fold_col).is_in(sorted(include_folds)))
        if exclude_folds is not None:
            lf = lf.filter(~pl.col(fold_col).is_in(sorted(exclude_folds)))

    exprs = []
    for c in num_cols:
        exprs += [pl.col(c).cast(pl.Float32).mean().alias(f"{c}_mean"),
                  pl.col(c).cast(pl.Float32).std(ddof=0).alias(f"{c}_std")]
    out = lf.select(exprs).collect(streaming=True)
    mean = out.select(
        [f"{c}_mean" for c in num_cols]).to_numpy().ravel().astype(np.float32)
    std = out.select(
        [f"{c}_std" for c in num_cols]).to_numpy().ravel().astype(np.float32)
    std[std == 0] = 1.0
    return mean, std


@dataclass(eq=False)
class TabNetCVTrainer:
    data_id: str
    train_paths: str | list[str]
    test_paths: str | list[str] | None = None

    features: Optional[list[str]] = None

    target: str = "target"
    fold_col: Optional[str] = None
    weight_col: Optional[str] = None
    cat_cols: Optional[list[str]] = None

    params: dict = field(default_factory=dict)

    n_fold: int = 5
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
            "n_d": 16,
            "n_a": 32,
            "n_steps": 5,
            "gamma": 1.5,
            "n_independent": 2,
            "n_shared": 2,
            "momentum": 0.3,
            "lambda_sparse": 1e-3,
            "eval_metric": ["logloss"],
            "patience": 5,
            "lr": 1e-3,
            "batch_size": 256,
            "virtual_batch_size": 32,
            "max_epochs": 100,
            "t_max": 50,
            "eta_min": 1e-6,
            "mask_type": "entmax",
            "device": "cuda"
        }

        self.params = {**default_params, **self.params}

        hdr = pl.read_parquet(self.train_paths, n_rows=0)
        all_cols = hdr.columns

        if self.fold_col is None:
            self.fold_col = f"{self.n_fold}fold-s{self.seed}"

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

            self.features = [
                c for c in all_cols
                if c not in meta and "fold" not in c
            ]

        if self.cat_cols is None:
            self.cat_cols = [
                c for c, dt in zip(hdr.columns, hdr.dtypes)
                if dt == pl.Categorical
            ]

        self.num_cols = [
            col for col in self.features
            if col not in self.cat_cols
        ]

        self.cat_idxs = [self.features.index(c) for c in self.cat_cols]
        self.num_idxs = [self.features.index(c) for c in self.num_cols]

        exprs = [pl.col(c).n_unique().alias(c) for c in self.cat_cols]
        df1 = self.lf_train.select(exprs).collect()

        if df1.width == 0 or df1.height == 0:
            self.cat_dims = []
        else:
            self.cat_dims = [int(x) if x is not None else 0 for x in df1.row(0)]

        self.embedding_dims = [
            min(50, (n + 1) // 2)
            for n in self.cat_dims
        ]

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
            "n_fold": self.n_fold,
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
            self.lf_test
                .select(pl.len())
                .collect()
                .item()
        )

        oof = np.zeros(train_rows, dtype=np.float32)
        test_pred = np.zeros(test_rows, dtype=np.float32)

        epoch_list = []
        loss_scores = []
        auc_scores = []

        for i in range(self.n_fold):
            print("=" * 22)
            print(f"===== Fold {i + 1} / {self.n_fold} =====")
            print("=" * 22)
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
                .astype(np.int64)
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
                .astype(np.int64)
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
                .collect(streaming=True)
                .to_numpy()
                .astype(np.float32)
            )

            mean, std = compute_feature_stats(
                self.train_paths,
                self.features,
                self.num_cols,
                self.fold_col,
                exclude_folds=[i]
            )

            X_train[:, self.num_idxs] = (X_train[:, self.num_idxs] - mean) / std
            X_valid[:, self.num_idxs] = (X_valid[:, self.num_idxs] - mean) / std
            test[:, self.num_idxs] = (test[:, self.num_idxs] - mean) / std

            history = {
                "train": {"loss": [], "auc": []},
                "valid": {"loss": [], "auc": []}
            }
            extra_hist = {"lr": []}

            model = TabNetClassifier(
                cat_idxs=self.cat_idxs,
                cat_dims=self.cat_dims,
                cat_emb_dim=self.embedding_dims,
                n_d=self.params["n_d"],
                n_a=self.params["n_a"],
                n_steps=self.params["n_steps"],
                gamma=self.params["gamma"],
                n_independent=self.params["n_independent"],
                n_shared=self.params["n_shared"],
                momentum=self.params["momentum"],
                lambda_sparse=self.params["lambda_sparse"],
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=self.params["lr"], weight_decay=1e-5),
                scheduler_params={
                    "T_max": self.params["t_max"],
                    "eta_min": self.params["eta_min"]
                },
                scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
                mask_type=self.params["mask_type"],
                verbose=1,
                seed=self.seed,
                device_name=self.params["device"]
            )

            model.fit(
                X_train=X_train,
                y_train=y_train.flatten(),
                eval_set=[(X_valid, y_valid.flatten())],
                eval_metric=self.params["eval_metric"],
                max_epochs=self.params["max_epochs"],
                patience=self.params["patience"],
                batch_size=self.params["batch_size"],
                virtual_batch_size=self.params["virtual_batch_size"],
                num_workers=0,
                drop_last=False
            )

            val_pred = model.predict_proba(X_valid)[:, 1]
            oof[val_idx] = val_pred
            test_pred += model.predict_proba(test)[:, 1]

            best_logloss = log_loss(y_valid, val_pred)
            loss_scores.append(best_logloss)
            print(f"Best Logloss: {best_logloss:.5f}")

            best_auc = roc_auc_score(y_valid, val_pred)
            auc_scores.append(best_auc)
            print(f"Best AUC: {best_auc:.5f}")

            t_fold_end = now()
            runtime = print_duration(
                t_fold_start, t_fold_end, f"Fold {i+1} Runtime"
            )
            fold_summary = {
                "loss": best_logloss,
                "auc": best_auc,
                "runtime": runtime
            }

            for lg in loggers:
                lg.on_fold_end(
                    i,
                    "epoch",
                    history,
                    extra_hist,
                    fold_summary
                )
            del model, X_train, y_train, X_valid, y_valid, test
            gc.collect()

        print("\n=== CV Results ===")
        y = (
            pl.read_parquet(self.train_paths, columns=self.target)
              .get_column(self.target)
              .cast(pl.Float32)
              .to_numpy()
        )
        oof_score_auc = roc_auc_score(y, oof)
        oof_score_loss = log_loss(y, oof)

        auc_mean = np.mean(auc_scores)
        auc_std = np.std(auc_scores)

        loss_mean = np.mean(loss_scores)
        loss_std = np.std(loss_scores)

        epoch_mean = np.mean(epoch_list)

        test_pred /= self.n_fold

        print(f"OOF AUC score: {oof_score_auc:.5f}")
        print(
            f"Mean: {auc_mean:.5f}, "
            f"Std: {auc_std:.5f}"
        )
        print(f"Avg best epoch: {epoch_mean}")

        t_total_end = now()
        total_runtime = print_duration(
            t_total_start, t_total_end, "Total CV Runtime"
        )
        print(f"Free CPU Mem: {round(free_ram_gib(), 2)} GB")
        print(f"Free GPU Mem: {round(free_vram_gib(), 2)} GB")

        result = CVResult(
            oof,
            test_pred,
            oof_score_auc
        )
        overall_summary = {
            "auc_oof": oof_score_auc,
            "auc_mean": auc_mean,
            "auc_std": auc_std,
            "loss_oof": oof_score_loss,
            "loss_mean": loss_mean,
            "loss_std": loss_std,
            "epoch_mean": epoch_mean,
            "total_runtime": total_runtime
        }
        for lg in loggers:
            lg.on_end(overall_summary)

        return result

    def fit_one_fold(
        self,
        fold_idx: int,
        loggers: list[CVLogger] | None = None
    ):
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
            .astype(np.int64)
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
            .astype(np.int64)
        )

        mean, std = compute_feature_stats(
            self.train_paths,
            self.features,
            self.num_cols,
            self.fold_col,
            exclude_folds=[fold_idx]
        )

        X_train[:, self.num_idxs] = (X_train[:, self.num_idxs] - mean) / std
        X_valid[:, self.num_idxs] = (X_valid[:, self.num_idxs] - mean) / std

        history = {
            "train": {"loss": [], "auc": []},
            "valid": {"loss": [], "auc": []}
        }
        extra_hist = {"lr": []}

        model = TabNetClassifier(
            cat_idxs=self.cat_idxs,
            cat_dims=self.cat_dims,
            cat_emb_dim=self.embedding_dims,
            n_d=self.params["n_d"],
            n_a=self.params["n_a"],
            n_steps=self.params["n_steps"],
            gamma=self.params["gamma"],
            n_independent=self.params["n_independent"],
            n_shared=self.params["n_shared"],
            momentum=self.params["momentum"],
            lambda_sparse=self.params["lambda_sparse"],
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=self.params["lr"], weight_decay=1e-5),
            scheduler_params={
                "T_max": self.params["t_max"],
                "eta_min": self.params["eta_min"]
            },
            scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
            mask_type=self.params["mask_type"],
            verbose=1,
            seed=self.seed,
            device_name=self.params["device"]
        )

        model.fit(
            X_train=X_train,
            y_train=y_train.flatten(),
            eval_set=[(X_valid, y_valid.flatten())],
            eval_metric=self.params["eval_metric"],
            max_epochs=self.params["max_epochs"],
            patience=self.params["patience"],
            batch_size=self.params["batch_size"],
            virtual_batch_size=self.params["virtual_batch_size"],
            num_workers=0,
            drop_last=False
        )

        val_pred = model.predict_proba(X_valid)[:, 1]

        best_logloss = log_loss(y_valid, val_pred)
        print(f"Best Logloss: {best_logloss:.5f}")

        best_auc = roc_auc_score(y_valid, val_pred)
        print(f"Best AUC: {best_auc:.5f}")

        del model, X_train, y_train, X_valid, y_valid
        gc.collect()

        t_total_end = now()
        total_runtime = print_duration(
            t_total_start, t_total_end, "Total CV Runtime"
        )
        fold_summary = {
            "loss": best_logloss,
            "auc": best_auc,
            "runtime": total_runtime
        }

        for lg in loggers:
            lg.on_fold_end(
                fold_idx,
                "epoch",
                history,
                extra_hist,
                fold_summary
            )
        print(f"Free CPU Mem: {round(free_ram_gib(), 2)} GB")
        print(f"Free GPU Mem: {round(free_vram_gib(), 2)} GB")

        return best_auc
