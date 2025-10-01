import os
import json
from pathlib import Path

import wandb
from optuna.exceptions import TrialPruned

from src.models.rfr.rfr_cv_trainer import RFRCVTrainer
from src.utils.snapshot_study import snapshot_study
from src.utils.telegram import send_message
from src.utils.loggers import WandbLogger

def create_objective(
    data_id: int,
    seed: int = 42,
    n_folds: int = 5,
    fold_idx: int = 0,
    wandb_project: str = "project",
    study_name: str = "study-rfr",
    opts: dict | None = None
):
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
        optuna_dir = Path("../../artifacts/optuna")
        try:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 150),
                "max_depth": trial.suggest_int("max_depth", 4, 30)
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
                tags=["rfr", level],
                reinit="finish_previous",
                dir="../../artifacts"
            )

            trainer = RFRCVTrainer(
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
                "score": float(score)
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
