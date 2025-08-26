from src.models.logreg.logreg_cv_trainer import LogRegCVTrainer


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
            "C": trial.suggest_float("C", 1e-2, 1e2, log=True)
        }

        trainer = LogRegCVTrainer(tr_df, params=params, n_splits=n_splits)

        score = trainer.fit_one_fold(fold=0)

        return score
    return objective