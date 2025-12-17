import math
from dataclasses import dataclass

import numpy as np
import polars as pl
import lightgbm as lgb

from src.models.base_cv_trainer import BaseCVTrainer, TrainResult


@dataclass
class LGBMCVTrainer(BaseCVTrainer):
    def __post_init__(self):
        super().__post_init__()
        self.log_axis_name = "iter"

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

        importances = model.feature_importance(importance_type="gain")
        total_gain = importances.sum()
        fi_df = pl.DataFrame(
            {
                "Feature": model.feature_name(),
                "Importance": [
                    ((v/total_gain)*100.0)/self.n_folds for v in importances
                ],
            }
        )

        return TrainResult(
                    model=model,
                    val_pred=model.predict(X_valid),
                    evals_result=evals_result,
                    fi=fi_df,
                    best_iteration=model.best_iteration
                )

    def predict_test(self, model):
        test = (
            self.lf_test
            .select(self.features)
            .collect(engine="streaming")
            .to_numpy()
            .astype(np.float32)
        )
        return model.predict(test)

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
        return TrainResult(
            model=model,
            val_pred=None,
            evals_result=None,
            extra=None,
            fi=None,
            best_iteration=model.best_iteration
        )