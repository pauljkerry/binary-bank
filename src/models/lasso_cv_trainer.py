from dataclasses import dataclass

import cudf
import cupy as cp
from cuml.linear_model import Lasso

from src.models.base_cv_trainer import BaseCVTrainer, TrainResult
from src.utils.compute_feature_stats import compute_feature_stats


@dataclass
class LassoCVTrainer(BaseCVTrainer):
    def __post_init__(self):
        super().__post_init__()
        self.log_axis_name = "iter"

        default_params = {
            "alpha": 1.0,
            "fit_intercept": True,
            "solver": "cd"
        }
        self.params = {**default_params, **self.params}

        # Cat colsを除外
        self.features = [c for c in self.features if c not in self.cat_cols]

        self.mean, self.std = compute_feature_stats(
            self.train_paths,
            self.features,
            self.features,
        )
        self.mean = cp.asarray(self.mean, dtype=cp.float32)
        self.std = cp.asarray(self.mean, dtype=cp.float32)

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

        model = Lasso(**self.params)
        model.fit(X_train, y_train)

        return TrainResult(
            model=model,
            val_pred=model.predict(X_valid).get(),
            evals_result=None,
            extra=None,
            fi=None,
            best_iteration=None
        )

    def predict_test(self, model):
        test = cudf.read_parquet(
            self.test_paths, columns=self.features
        ).to_cupy()

        test -= self.mean
        test /= (self.std + 1e-8)
        return model.predict(test).get()

    def train_on_all_data(self):
        pass
