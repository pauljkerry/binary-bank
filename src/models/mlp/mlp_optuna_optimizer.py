import optuna
from src.models.mlp.mlp_cv_trainer import MLPCVTrainer


def create_objective(
    tr_df,
    n_splits=5,
    epochs=100,
    early_stopping_rounds=20,
    min_epochs=30,
    use_gpu=True,
):
    """
    Optunaの目的関数（objective）を生成する関数。

    Parameters
    ----------
    tr_df : pd.DataFrame
        訓練データ。
    n_splits : int, default 5
        CV分割数。
    epochs : int, default 100
        エポック数。
    early_stopping_rounds : int, default 200
        早期停止ラウンド数。
    min_epochs : int, default 50
        最低限学習するエポック数
    use_gpu : bool, default True
        Trueの場合はGPUが使用可能であれば使用する。

    Returns
    -------
    function
        Optunaで使用する目的関数。
    """
    def objective(trial):
        num_layers = trial.suggest_int("num_layers", 2, 4)

        hidden_dim1 = trial.suggest_int("hidden_dim1", 256, 1024, step=32)
        hidden_dim2 = trial.suggest_int(
            "hidden_dim2", 128, hidden_dim1, step=32)
        hidden_dim3 = trial.suggest_int(
            "hidden_dim3", 64, hidden_dim2, step=32
        ) if num_layers >= 3 else None
        hidden_dim4 = trial.suggest_int(
            "hidden_dim4", 32, hidden_dim3, step=32
        ) if num_layers >= 4 else None

        hidden_dims = [hidden_dim1, hidden_dim2]
        if num_layers >= 3:
            hidden_dims.append(hidden_dim3)
        if num_layers >= 4:
            hidden_dims.append(hidden_dim4)

        params = {
            "n_splits": n_splits,
            "epochs": epochs,
            "early_stopping_rounds": early_stopping_rounds,
            "min_epochs": min_epochs,
            "use_gpu": use_gpu,
            "batch_size": trial.suggest_int(
                "batch_size", 256, 2048, step=32
            ),
            "lr": trial.suggest_float("lr", 1e-3, 1e-1, log=True),
            "eta_min": trial.suggest_float("eta_min", 1e-4, 1e-3, log=True),
            "dropout_rate": round(trial.suggest_float(
                "dropout_rate", 0.1, 0.4, step=0.05), 2),
            "activation": trial.suggest_categorical("activation", [
                "ReLU",
                "LeakyReLU",
                "GELU",
                "SiLU",
                # "Tanh",
                # "ELU",
                # "Sigmoid"
            ]),
            "hidden_dims": hidden_dims,
        }

        trainer = MLPCVTrainer(**params)
        trainer.fit_one_fold(tr_df, fold=0)

        return trainer.fold_scores[0]
    return objective


def run_optuna_search(
    objective, n_trials=50, n_jobs=1,
    direction="minimize", study_name="mlp_study", storage=None,
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
    study_name : str or None, default "mlp_study"
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