import optuna
from src.utils.telegram import send_message


def run_optuna_search(
    objective,
    n_trials: int = 50,
    direction: str = "minimize",
    study_name: str = "study",
    storage: str = None,
    initial_params: dict | list[dict] = None,
    sampler: optuna.samplers.BaseSampler | None = None,
    pruner=None
) -> optuna.Study:
    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=sampler or optuna.samplers.TPESampler(),
        pruner=pruner
    )

    if initial_params is not None:
        if isinstance(initial_params, dict):
            study.enqueue_trial(initial_params)
        elif isinstance(initial_params, list):
            for p in initial_params:
                if not isinstance(p, dict):
                    raise ValueError(
                        "Each element of initial_params must be a dict."
                    )
                study.enqueue_trial(p)
                print(f"[initial] {len(initial_params)} trial(s) will be enqueued:")
        else:
            raise ValueError("initial_params must be a dict or list[dict].")
    else:
        print("[initial] none")

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True
    )

    msg = (
        "Training Complete!\n"
        f"Study: {study.study_name}\n"
        f"Best Value: {study.best_value:.5f}\n"
        f"Trials: {n_trials}"
    )
    send_message(msg)
    return study