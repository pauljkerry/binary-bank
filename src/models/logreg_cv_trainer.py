import re
from dataclasses import dataclass

import cudf
import cupy as cp
import numpy as np
import polars as pl
from cuml.linear_model import LogisticRegression

from src.models.base_cv_trainer import BaseCVTrainer


def compute_feature_stats(
    paths: list[str],
    features: list[str],
    num_cols: list[str]
) -> (float, float):
    lf = pl.scan_parquet(paths, low_memory=True)

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
    return cp.asarray(mean, dtype=cp.float32), cp.asarray(std, dtype=cp.float32)


@dataclass
class LogRegCVTrainer(BaseCVTrainer):
    def __post_init__(self):
        super().__post_init__()
        self.log_axis_name = "iter"

        default_params = {
            "C": 1.0,
            "penalty": "l2",
            "solver": "qn",
            "max_iter": 3000,
            "class_weight": None
        }
        self.params = {**default_params, **self.params}

        if self.features is None:
            meta = {
                c
                for c in ("row_id", self.target, self.weight_col, self.fold_col)
                if c and c in self.all_cols
            }
            if self.cat_cols:
                meta |= self.cat_cols
            pat = re.compile(r"^\d+fold(?:-[A-Za-z0-9]+)?$")
            self.features = [
                c for c in self.all_cols
                if c not in meta and not pat.fullmatch(c)
            ]

        self.mean, self.std = compute_feature_stats(
            self.train_paths,
            self.features,
            self.features,
        )

    def train_model(self, fold):
        train = cudf.read_parquet(
            self.train_paths,
            columns=self.features + [self.target, self.fold_col]
        )

        X_train = (
            train[train[self.fold_col] != fold]
            [self.features].to_cupy().astype(cp.float32)
        )
        y_train = (
            train[train[self.fold_col] != fold]
            [self.target].to_cupy().astype(cp.float32)
        )

        X_valid = (
            train[train[self.fold_col] == fold]
            [self.features].to_cupy().astype(cp.float32)
        )

        X_train -= self.mean
        X_train /= (self.std + 1e-8)

        X_valid -= self.mean
        X_valid /= (self.std + 1e-8)

        model = LogisticRegression(**self.params)
        model.fit(X_train, y_train)

        pred = model.predict_proba(X_valid).get()[:, 1]
        return model, pred, None, None

    def predict_test(self, model):
        test = cudf.read_parquet(
            self.test_paths, columns=self.features
        ).to_cupy()

        test -= self.mean
        test /= (self.std + 1e-8)
        pred = model.predict_proba(test).get()[:, 1]
        return pred
