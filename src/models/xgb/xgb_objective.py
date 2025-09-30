import os
import json
from pathlib import Path

import wandb
from optuna.exceptions import TrialPruned

from src.models.xgb.xgb_cv_trainer import XGBCVTrainer
from src.utils.snapshot_study import snapshot_study
from src.utils.telegram import send_message
from src.utils.loggers import WandbLogger


def create_objective(
    data_id: int,
    seed: int = 42,
    n_folds: int = 5,
    fold_idx: int = 0,
    wandb_project: str = "project",
    study_name: str = "study-xgb",
    opts: dict | None = None
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
    early_stopping_rounds: Optional[int] = None

    Returns
    -------
    function
        Optunaで使用する目的関数。
    """

    def objective(trial):
        optuna_dir = Path("../../artifacts/optuna")
        try:
            params = {
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    0.02,
                    0.02
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    6,
                    13
                ),
                "min_child_weight": trial.suggest_float(
                    "min_child_weight",
                    0,
                    100
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree",
                    0.3,
                    0.7
                ),
                "subsample": trial.suggest_float(
                    "subsample",
                    0.5,
                    0.9
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha",
                    1e-4,
                    40.0,
                    log=True
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda",
                    1e-4,
                    10.0,
                    log=True
                ),
            }

            with open(f"../../artifacts/features/{data_id}/meta.json")as f:
                m = json.load(f)

            train_paths = m["train_paths"]
            level = m["level"]

            run = wandb.init(
                project=wandb_project,
                group=study_name,
                name=f"trl{trial.number}",
                job_type="optuna-search",
                config={
                    "data_id": data_id,
                    "n_folds": n_folds,
                    **params,
                    **opts
                },
                tags=["xgb", level],
                reinit="finish_previous",
                dir="../../artifacts"
            )

            trainer = XGBCVTrainer(
                data_id,
                train_paths,
                n_folds=n_folds,
                params=params,
                seed=seed,
                opts=opts
            )

            score = trainer.fit_one_fold(
                fold_idx,
                loggers=[WandbLogger(run=run)]
            )

            os.makedirs(optuna_dir / f"{study_name}", exist_ok=True)
            path = optuna_dir / f"{study_name}/trl{trial.number}.json"
            manifest = {
                "params": params,
                "n_folds": n_folds,
                "seed": seed,
                "fold_idx": fold_idx,
                "wandb_id": run.id,
                "wandb_url": run.url,
                "opts": opts,
                "score": score
            }
            with open(path, "w") as f:
                json.dump(manifest, f, indent=4)

            return score
        except RuntimeError as e:
            msg = str(e)
            if "CUDA out of memory" in msg:
                send_message(
                    f"[OOM] study={study_name} tr={trial.number} "
                    f"params={trial.params}"
                )
                raise TrialPruned("OOM -> pruned")
            else:
                send_message(
                    f"[ERROR] study={study_name} tr={trial.number} "
                    f"{type(e).__name__}: {msg}"
                )
                raise
        finally:
            wandb.finish()
            try:
                N = 10
                if (trial.number+1) % N == 0 and trial.number != 0:
                    _ = snapshot_study(
                        study=trial.study,
                        study_name=study_name,
                        trial_num=trial.number,
                        out_root=optuna_dir,
                        send_telegram=True,
                    )
            except Exception:
                pass

    return objective
