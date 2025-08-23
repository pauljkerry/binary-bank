from src.models.rfc.rfc_cv_trainer import RFCCVTrainer


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
            "n_estimators": trial.suggest_int("n_estimators", 50, 150),
            "max_depth": trial.suggest_int("max_depth", 4, 100)
        }

        trainer = RFCCVTrainer(
            params=params,
            n_splits=n_splits
        )

        trainer.fit_one_fold(tr_df, fold=0)

        return trainer.fold_scores[0]
    return objective