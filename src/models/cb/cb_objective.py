from src.models.cb.cb_cv_trainer import CBCVTrainer


def create_objective(
    tr_df,
    n_splits=5,
    early_stopping_rounds=200,
    n_jobs=1,
    task_type="GPU"
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
        CatBoost並列数。
    task_type : str, default "GPU"
        使用する計算資源。

    Returns
    -------
    function
        Optunaで使用する目的関数。
    """
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.02),
            "depth": trial.suggest_int("depth", 6, 16),
            # "rsm": trial.suggest_float("rsm", 0.2, 0.4),
            # "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "min_data_in_leaf": trial.suggest_float(
                "min_data_in_leaf", 10, 100),
            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg", 1e-2, 20.0
            ),
            "bagging_temperature": trial.suggest_float(
                "bagging_temperature", 1e-2, 1.0
            ),
            "random_strength": trial.suggest_int(
                "random_strength", 1, 80
            ),
            "border_count": trial.suggest_int(
                "border_count", 128, 255
            ),
            "task_type": task_type,
            "early_stopping_rounds": early_stopping_rounds,
        }

        trainer = CBCVTrainer(
            tr_df, params=params, n_splits=n_splits,
            early_stopping_rounds=early_stopping_rounds)

        score = trainer.fit_one_fold()

        return score
    return objective