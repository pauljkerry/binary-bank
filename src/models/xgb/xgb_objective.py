from src.models.xgb.xgb_cv_trainer import XGBCVTrainer


def create_objective(
    tr_df,
    n_splits=5,
    early_stopping_rounds=200,
    n_jobs=1
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
    n_jobs : int, default 1
        XGB並列数。

    Returns
    -------
    function
        Optunaで使用する目的関数。
    """
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.03),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 0, 100),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.3, 0.6),
            "subsample": trial.suggest_float("subsample",
                                             0.5, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha",
                                             1e-4, 40.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda",
                                              1e-4, 10.0, log=True),
            "n_jobs": n_jobs
        }

        trainer = XGBCVTrainer(
            tr_df, n_splits=n_splits,
            params=params,
            early_stopping_rounds=early_stopping_rounds
        )

        score = trainer.fit_one_fold(fold=0)

        return score
    return objective