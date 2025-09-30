import os
import json
from pathlib import Path

import wandb
from optuna.exceptions import TrialPruned

from src.models.tabnet.tabnet_cv_trainer import TabNetCVTrainer
from src.utils.snapshot_study import snapshot_study
from src.utils.telegram import send_message
from src.utils.loggers import WandbLogger


def create_objective(
    data_id: int,
    seed: int = 42,
    n_fold: int = 5,
    fold_idx: int = 0,
    wandb_project: str = "project",
    study_name: str = "study-tabnet",
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
                "n_d": trial.suggest_int(
                    "n_d", 8, 24
                ),
                "n_a": trial.suggest_int(
                    "n_a", 8, 24
                ),
                "n_steps": trial.suggest_int(
                    "n_steps", 1, 10
                ),
                "gamma": trial.suggest_float(
                    "gamma", 1.2, 2.0
                ),
                "n_independent": trial.suggest_int(
                    "n_independent", 1, 4
                ),
                "n_shared": trial.suggest_int(
                    "n_shared", 1, 4
                ),
                "momentum": trial.suggest_float(
                    "momentum", 0.02, 0.4
                ),
                "lambda_sparse": trial.suggest_float(
                    "lambda_sparse", 1e-5, 1e-3, log=True
                ),
                "lr": trial.suggest_float(
                    "lr", 1e-4, 1e-3),
                "batch_size": trial.suggest_int(
                    "batch_size", 5240, 10480, step=32),
                "eta_min": trial.suggest_float(
                    "eta_min", 1e-4, 1e-3, log=True
                ),
                "mask_type": trial.suggest_categorical(
                    "mask_type", ["entmax", "sparsemax"]
                )
            }
            params = {**params, **opts}
            params["virtual_batch_size"] = params["batch_size"] / 8

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
                    "n_fold": n_fold,
                    **params,
                    **opts
                },
                tags=["tabnet", level],
                reinit="finish_previous",
                dir="../../artifacts"
            )

            trainer = TabNetCVTrainer(
                data_id,
                train_paths,
                n_fold=n_fold,
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
                "n_fold": n_fold,
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
                    f"[OOM] study={study_name} tr={trial.number} params={trial.params}"
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
