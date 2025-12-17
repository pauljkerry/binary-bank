from dataclasses import dataclass

import numpy as np
import polars as pl
from catboost import CatBoostClassifier, Pool

from src.models.base_cv_trainer import BaseCVTrainer, TrainResult


@dataclass(eq=False)
class CBCVTrainer(BaseCVTrainer):
    def __post_init__(self):
        super().__post_init__()
        self.log_axis_name = "iter"

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

    def train_model(self, fold) -> TrainResult:
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
            train_pool,
            eval_set=valid_pool,
            use_best_model=True
        )

        return TrainResult(
                    model=model,
                    val_pred=model.predict_proba(X_valid)[:, 1],
                    evals_result=None,
                    fi=None,
                    best_iteration=model.best_iteration_
                )

    def predict_test(self, model):
        test = (
            self.lf_test
            .select(self.features)
            .collect(engine="streaming")
            .to_numpy()
            .astype(np.float32)
        )
        return model.predict_proba(test)[:, 1]

    def train_on_all_data(self) -> TrainResult:
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

        return TrainResult(
                    model=model,
                    val_pred=None,
                    evals_result=None,
                    fi=None,
                    best_iteration=model.best_iteration_
                )
