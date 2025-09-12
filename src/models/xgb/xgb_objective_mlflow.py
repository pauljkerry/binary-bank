import mlflow
from src.models.xgb.xgb_cv_trainer_mlflow import XGBCVTrainer


def create_objective(
    data_id,
    base_dir,
    seed=42,
    n_fold=5,
    fold=0,
    early_stopping_rounds=200,
    n_jobs=1,
    study_name="study-xgb",
    tracking_uri=None
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
        XGB並列数。

    Returns
    -------
    function
        Optunaで使用する目的関数。
    """
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.03),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 0, 100),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.3, 0.6),
            "subsample": trial.suggest_float("subsample",
                                             0.5, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha",
                                             1e-4, 40.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda",
                                              1e-4, 10.0, log=True),
            "n_jobs": n_jobs
        }

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(study_name)
        mlflow.enable_system_metrics_logging()

        with mlflow.start_run(run_name=f"tr{trial.number}") as run:
            mlflow.set_tags({
                "optuna-trial": trial.number,
                "data_id": data_id,
                "fold_used": fold,
                "seed": seed,
                "n_fold": n_fold,
            })
            mlflow.log_params(
                {"seed": 42,
                 "data_id": "026",
                 "fold_used": "fold0",
                 **params})

            trainer = XGBCVTrainer(
                data_id, base_dir, n_fold,
                params=params, seed=seed,
                early_stopping_rounds=early_stopping_rounds
            )

            score = trainer.fit_one_fold(fold)
            mlflow.log_metric("auc", score)

            trial.set_user_attr("data_id", data_id)
            trial.set_user_attr("seed", seed)
            trial.set_user_attr("n_fold", n_fold)
            trial.set_user_attr("fold_used", fold)

        return score
    return objective