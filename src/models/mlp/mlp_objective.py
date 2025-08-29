from src.models.mlp.mlp_cv_trainer import MLPCVTrainer


def create_objective(
    tr_df,
    n_splits=5,
    max_epochs=30,
    early_stopping_rounds=5,
    min_epochs=10,
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
    max_epochs : int, default 100
        エポック数。
    early_stopping_rounds : int, default 20
        早期停止ラウンド数。
    min_epochs : int, default 10
        最低限学習するエポック数
    use_gpu : bool, default True
        Trueの場合はGPUが使用可能であれば使用する。

    Returns
    -------
    function
        Optunaで使用する目的関数。
    """
    def objective(trial):
        num_layers = trial.suggest_int("num_layers", 1, 4)

        hidden_dim1 = trial.suggest_int("hidden_dim1", 256, 1024, step=32)

        if num_layers >= 2:
            hidden_dim2 = trial.suggest_int("hidden_dim2", 128, hidden_dim1, step=32)
        else:
            hidden_dim2 = -1
            trial.suggest_int("hidden_dim2", -1, -1)

        if num_layers >= 3:
            hidden_dim3 = trial.suggest_int("hidden_dim3", 64, hidden_dim2, step=32)
        else:
            hidden_dim3 = -1
            trial.suggest_int("hidden_dim3", -1, -1)

        if num_layers >= 4:
            hidden_dim4 = trial.suggest_int("hidden_dim4", 32, hidden_dim3, step=32)
        else:
            hidden_dim4 = -1
            trial.suggest_int("hidden_dim4", -1, -1)

        params = {
            "hidden_dim1": hidden_dim1,
            "hidden_dim2": hidden_dim2,
            "hidden_dim3": hidden_dim3,
            "hidden_dim4": hidden_dim4,
            "n_splits": n_splits,
            "max_epochs": max_epochs,
            "early_stopping_rounds": early_stopping_rounds,
            "min_epochs": min_epochs,
            "use_gpu": use_gpu,
            "batch_size": trial.suggest_int(
                "batch_size", 512, 1120, step=32
            ),
            "lr": trial.suggest_float("lr", 1e-3, 1e-1, log=True),
            "lr_min": trial.suggest_float("lr_min", 1e-4, 1e-3, log=True),
            "dropout_rate": round(trial.suggest_float(
                "dropout_rate", 0.1, 0.6, step=0.05), 2),
            "activation": trial.suggest_categorical("activation", [
                "ReLU",
                "LeakyReLU",
                "GELU",
                "SiLU",
                # "Tanh",
                # "ELU",
                # "Sigmoid"
            ]),
        }
        trainer = MLPCVTrainer(tr_df, params=params, n_splits=n_splits)
        score = trainer.fit_one_fold(fold=0)

        return score
    return objective