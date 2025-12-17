from dataclasses import dataclass
import polars as pl
from pytabkit import RealMLP_TD_Classifier

from src.models.base_cv_trainer import BaseCVTrainer, TrainResult


@dataclass
class RealMLPCVTrainer(BaseCVTrainer):
    def __post_init__(self):
        super().__post_init__()
        self.log_axis_name = "epoch"

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

    def train_model(self, fold):
        train = pl.read_parquet(
            self.train_paths,
            columns=self.features + [self.target, self.fold_col]
        )

        X_train = train[train[self.fold_col] != fold][self.features].to_numpy()
        y_train = train[train[self.fold_col] != fold][self.target].to_numpy()

        X_valid = train[train[self.fold_col] == fold][self.features].to_numpy()

        model = RealMLP_TD_Classifier(**self.params)
        model.fit(X_train, y_train)

        return TrainResult(
            model=model,
            val_pred=model.predict(X_valid),
            evals_result=None,
            extra=None,
            fi=None,
            best_iteration=None
        )

    def predict_test(self, model):
        test = pl.read_parquet(self.test_paths, columns=self.features)

        return model.predict(test)

    def train_on_all_data(self):
        pass
