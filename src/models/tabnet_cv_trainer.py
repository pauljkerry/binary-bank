from dataclasses import dataclass

import torch
import numpy as np
import polars as pl

from pytorch_tabnet.tab_model import TabNetClassifier

from src.models.base_cv_trainer import BaseCVTrainer, TrainResult
from src.utils.compute_feature_stats import compute_feature_stats


@dataclass(eq=False)
class TabNetCVTrainer(BaseCVTrainer):
    def __post_init__(self):
        super().__post_init__()
        self.log_axis_name = "epoch"

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
            "max_epochs": 100,
            "t_max": 50,
            "eta_min": 1e-6,
            "mask_type": "entmax",
            "device": "cuda"
        }

        self.params = {**default_params, **self.params}
        self.params["virtual_batch_size"] = self.params["batch_size"] / 8

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

        self.mean, self.std = compute_feature_stats(
            self.train_paths,
            self.features,
            self.num_cols
        )

    def train_model(self, fold):
        need_cols = self.features + [self.target, "row_id"]
        train = (
            self.lf_train
            .filter(pl.col(self.fold_col) != fold)
            .select(need_cols)
            .collect(engine="streaming")
        )
        valid = (
            self.lf_train
            .filter(pl.col(self.fold_col) == fold)
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

        X_train[:, self.num_idxs] = (
            (X_train[:, self.num_idxs] - self.mean)
            / self.std
        )
        X_valid[:, self.num_idxs] = (
            (X_valid[:, self.num_idxs] - self.mean)
            / self.std
        )

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

        return TrainResult(
            model=model,
            val_pred=model.predict_proba(X_valid)[:, 1],
            evals_result=None,
            extra=None,
            fi=None,
            best_iteration=None
        )

    def predict_test(self, model):
        test = (
            self.lf_test
            .select(self.features)
            .collect(streaming=True)
            .to_numpy()
            .astype(np.float32)
        )
        test[:, self.num_idxs] = (
            (test[:, self.num_idxs] - self.mean)
            / self.std
        )

        return model.predict_proba(test)[:, 1]

    def train_on_all_data(self):
        pass
