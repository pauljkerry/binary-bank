import os
import json
import wandb
from optuna.exceptions import TrialPruned

from src.models.xgb.xgb_cv_trainer import XGBCVTrainer
from src.utils.snapshot_study import snapshot_study
from src.utils.telegram import send_message
from src.utils.logging import WandbLogger


def create_objective(
    data_id: int,
    feature_dir: str,
    seed: int = 42,
    n_fold: int = 5,
    fold_idx: int = 0,
    early_stopping_rounds: int = 200,
    batch_rows: int = 200_000,
    n_jobs: int = 1,
    wandb_project: str = "project",
    study_name: str = "study-xgb",
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
        run = wandb.init(
            project=wandb_project,
            group=study_name,
            name=f"trl{trial.number}",
            config={
                "data_id": data_id,
                "n_fold": n_fold
            },
            tags=["xgb", "optuna"],
            reinit=True,
        )
        try:
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.02),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_float("min_child_weight", 0, 100),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.6),
                "subsample": trial.suggest_float("subsample", 0.5, 0.9),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 40.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "n_jobs": n_jobs,
                "early_stopping_rounds": early_stopping_rounds
            }

            wandb.config.update(params)

            trainer = XGBCVTrainer(
                data_id,
                feature_dir,
                n_fold=n_fold,
                params=params,
                seed=seed,
                batch_rows=batch_rows
            )

            score = trainer.fit_one_fold(
                fold_idx,
                loggers=[WandbLogger(run=run)]
            )

            wandb.finish()

            os.makedirs(f"../artifacts/params/{study_name}", exist_ok=True)
            path = f"../artifacts/params/{study_name}/trl{trial.number}.json"
            with open(path, "w") as f:
                json.dump(params, f, indent=4)

            return score
        except RuntimeError as e:
            wandb.finish()
            msg = str(e)
            if "CUDA out of memory" in msg:
                send_message(
                    f"[OOM] study={study_name} tr={trial.number} params={trial.params}"
                )
                raise TrialPruned("OOM -> pruned")
            else:
                send_message(
                    f"[ERROR] study={study_name} tr={trial.number} {type(e).__name__}: {msg}"
                )
                raise
        finally:
            try:
                N = 10  # 周期
                if trial.number % N == 0:
                    _ = snapshot_study(
                        study=trial.study,
                        study_name=study_name,
                        trial_num=trial.number,
                        out_root="runs/optuna",
                        send_telegram=True,
                    )
            except Exception:
                pass
            try:
                import cupy as cp

                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass
            import ctypes
            import gc

            gc.collect()
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass

    return objective
