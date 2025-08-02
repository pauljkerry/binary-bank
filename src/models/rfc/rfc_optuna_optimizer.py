import optuna
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


def run_optuna_search(
    objective, n_trials=50, direction="minimize", study_name="rfc_study",
    storage=None, initial_params: dict = None, sampler=None
):
    """
    Optunaによるハイパーパラメータ探索を実行する関数。

    Parameters
    ----------
    objective : function
        Optunaの目的関数。
    n_trials : int, default 50
        試行回数。
    direction : str, default "minimize"
        Optunaの探索方向。
    study_name : str or None, default "lgbm_study"
        StudyName。
    storage : str or None, default None
        保存先URL。
    initial_params : dict or None, default None
        初期の試行パラメータ。
    sampler : optuna.samplers.BaseSampler or None, default TPESampler
        使用するSampler。

    Returns
    -------
    study : optuna.Study
        探索結果のStudyオブジェクト。
    """
    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=sampler or optuna.samplers.TPESampler()
    )

    if initial_params is not None:
        study.enqueue_trial(initial_params)

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True
    )

    return study