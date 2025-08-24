import math
from src.models.lgbm.lgbm_cv_trainer import LGBMCVTrainer


def create_objective(
    tr_df,
    n_splits=5,
    early_stopping_rounds=200,
    n_jobs=20
):
    """
    Optunaの目的関数（objective）を生成する関数。

    Parameters
    ----------
    tr_df : pd.DataFrame
        訓練データ。
    n_splits : int, default 5
        CV分割数。
    early_stopping_rounds : int, default 200
        EarlyStoppingのラウンド数。
    n_jobs: int, default 20
        LGBM並列数。

    Returns
    -------
    objective : function
        optunaで使用する目的関数。
    """
    trainer = LGBMCVTrainer(
        tr_df, n_splits=n_splits,
        early_stopping_rounds=early_stopping_rounds
    )

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.02, 0.02),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "num_leaves": trial.suggest_int("num_leaves", 300, 1200),
            "min_child_samples": trial.suggest_int("min_child_samples",
                                                   100, 20000),
            "min_split_gain": trial.suggest_float("min_split_gain",
                                                  1e-5, 10, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction",
                                                    0.3, 1.00),
            "bagging_fraction": trial.suggest_float("bagging_fraction",
                                                    0.65, 1.00),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 15),
            "lambda_l1": trial.suggest_float("lambda_l1",
                                             1e-5, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2",
                                             1e-5, 10.0, log=True),
            "n_jobs": n_jobs,
        }

        min_required_depth = int(math.log2(params["num_leaves"])) + 1
        params["max_depth"] = max(params["max_depth"], min_required_depth)

        trainer.params = params

        score = trainer.fit_one_fold(fold=0)

        return score
    return objective