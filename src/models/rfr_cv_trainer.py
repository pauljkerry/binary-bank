from dataclasses import dataclass
import cudf
from cuml.ensemble import RandomForestRegressor

from src.models.base_cv_trainer import BaseCVTrainer, TrainResult


@dataclass
class RFRCVTrainer(BaseCVTrainer):
    def __post_init__(self):
        super().__post_init__()
        self.log_axis_name = "iter"

        default_params = {
            "n_estimators": 100,
            "max_depth": 16,
            "bootstrap": True,
            "random_state": self.seed,
            "n_streams": 1
        }

        self.params = {**default_params, **self.params}

    def train_model(self, fold):
        train = cudf.read_parquet(
            self.train_paths,
            columns=self.features + [self.target, self.fold_col]
        )

        X_train = train[train[self.fold_col] != fold][self.features].to_cupy()
        y_train = train[train[self.fold_col] != fold][self.target].to_cupy()

        X_valid = train[train[self.fold_col] == fold][self.features].to_cupy()

        model = RandomForestRegressor(**self.params)
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

        return model.predict(test).get()

    def train_on_all_data(self):
        pass
