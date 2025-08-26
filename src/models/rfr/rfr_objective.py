from src.models.rfr.rfr_cv_trainer import RFRCVTrainer


def create_objective(tr_df, n_splits=5):
    """
    Optunaの目的関数（objective）を生成する関数。

    Parameters
    ----------
    tr_df : cudf.DataFrame
        訓練データ。
    n_splits : int, default 5
        CV分割数。

    Returns
    -------
    objective : function
        optunaで使用する目的関数。
    """
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 100),
            "max_depth": trial.suggest_int("max_depth", 4, 20)
        }

        trainer = RFRCVTrainer(tr_df, params=params, n_splits=n_splits)

        score = trainer.fit_one_fold(fold=0)

        return score
    return objective