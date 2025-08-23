from src.models.ridge.ridge_cv_trainer import RidgeCVTrainer


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
            "alpha": trial.suggest_float("alpha", 1e-2, 1e2, log=True)
        }

        trainer = RidgeCVTrainer(
            params=params,
            n_splits=n_splits
        )

        trainer.fit_one_fold(tr_df, fold=0)

        return trainer.fold_scores[0]
    return objective