import optuna
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
            "depth": trial.suggest_int("depth", 3, 16),
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
            params=params,
            n_splits=n_splits,
            early_stopping_rounds=early_stopping_rounds
        )

        trainer.fit_one_fold(tr_df, fold=0)

        return trainer.fold_scores[0]
    return objective


def run_optuna_search(
    objective, n_trials=50, n_jobs=1,
    direction="minimize", study_name="cb_study", storage=None,
    initial_params: dict = None, sampler=None
):
    """
    Optunaによるハイパーパラメータ探索を実行する関数。

    Parameters
    ----------
    objective : function
        Optunaの目的関数。
    n_trials : int, default 50
        試行回数。
    n_jobs : int, default 1
        並列実行数。
    direction : str, default "minimize"
        Optunaの探索方向。
    study_name : str or None, default "xgb_study"
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
        n_jobs=n_jobs,
        show_progress_bar=True
    )

    return study