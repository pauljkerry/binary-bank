import os
import json
from pathlib import Path

import wandb
from optuna.exceptions import TrialPruned

from src.models.mlp.mlp_cv_trainer import MLPCVTrainer
from src.utils.snapshot_study import snapshot_study
from src.utils.telegram import send_message
from src.utils.loggers import WandbLogger


def create_objective(
    data_id: int,
    seed: int = 42,
    n_folds: int = 5,
    fold_idx: int = 0,
    wandb_project: str = "project",
    study_name: str = "study-mlp",
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
            num_layers = trial.suggest_int("num_layers", 1, 4)

            hidden_dim1 = trial.suggest_int(
                "hidden_dim1", 256, 1024, step=32
            )

            if num_layers >= 2:
                hidden_dim2 = trial.suggest_int(
                    "hidden_dim2", 128, hidden_dim1, step=32
                )
            else:
                hidden_dim2 = -1
                trial.suggest_int(
                    "hidden_dim2", -1, -1
                )

            if num_layers >= 3:
                hidden_dim3 = trial.suggest_int(
                    "hidden_dim3", 64, hidden_dim2, step=32
                )
            else:
                hidden_dim3 = -1
                trial.suggest_int(
                    "hidden_dim3", -1, -1
                )

            if num_layers >= 4:
                hidden_dim4 = trial.suggest_int(
                    "hidden_dim4", 32, hidden_dim3, step=32
                )
            else:
                hidden_dim4 = -1
                trial.suggest_int("hidden_dim4", -1, -1)

            params = {
                "hidden_dim1": hidden_dim1,
                "hidden_dim2": hidden_dim2,
                "hidden_dim3": hidden_dim3,
                "hidden_dim4": hidden_dim4,
                "batch_size": trial.suggest_int(
                    "batch_size", 512, 1120, step=32
                ),
                "lr": trial.suggest_float(
                    "lr", 1e-3, 1e-1, log=True
                ),
                "eta_min": trial.suggest_float(
                    "eta_min", 1e-4, 1e-3, log=True
                ),
                "dropout_rate": round(trial.suggest_float(
                    "dropout_rate", 0.1, 0.6, step=0.05), 2),
                "activation": trial.suggest_categorical(
                    "activation", [
                        "ReLU",
                        "LeakyReLU",
                        "GELU",
                        "SiLU"
                    ]
                ),
            }
            params = {**params, **opts}

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
                tags=["mlp", level],
                reinit="finish_previous",
                dir="../../artifacts"
            )

            trainer = MLPCVTrainer(
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
