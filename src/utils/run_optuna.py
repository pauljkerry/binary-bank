import optuna
from src.utils.telegram import send_telegram_message


def run_optuna_search(
    objective, n_trials=50, n_jobs=1,
    direction="minimize", study_name="study", storage=None,
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
    study_name : str or None, default "study"
        StudyName。
    storage : str or None, default None
        保存先URL。
    initial_params : dict, list[dict] or None, default None
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
        if isinstance(initial_params, dict):
            study.enqueue_trial(initial_params)
        elif isinstance(initial_params, list):
            for p in initial_params:
                if not isinstance(p, dict):
                    raise ValueError("initial_paramsの各要素はdictである必要があります。")
                study.enqueue_trial(p)
        else:
            raise ValueError("initial_paramsはdictまたはlist[dict]である必要があります。")

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True
    )

    msg = (
        "Training Complete!\n"
        f"Study: {study.study_name}\n"
        f"Best Value: {study.best_value:.5f}\n"
        f"Trials: {n_trials}"
    )
    send_telegram_message(msg)
    return study