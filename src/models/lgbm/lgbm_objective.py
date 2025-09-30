import os
import json
import math
from pathlib import Path

import wandb
from optuna.exceptions import TrialPruned

from src.models.lgbm.lgbm_cv_trainer import LGBMCVTrainer
from src.utils.snapshot_study import snapshot_study
from src.utils.telegram import send_message
from src.utils.loggers import WandbLogger


def create_objective(
    data_id: int,
    seed: int = 42,
    n_folds: int = 5,
    fold_idx: int = 0,
    wandb_project: str = "project",
    study_name: str = "study-lgbm",
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
    n_jobs: int, default 20
        LGBM並列数。

    Returns
    -------
    objective : function
        optunaで使用する目的関数。
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
                    5,
                    20
                ),
                "num_leaves": trial.suggest_int(
                    "num_leaves",
                    300,
                    1200
                ),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples",
                    100,
                    20000
                ),
                "min_split_gain": trial.suggest_float(
                    "min_split_gain",
                    1e-5,
                    10,
                    log=True
                ),
                "feature_fraction": trial.suggest_float(
                    "feature_fraction",
                    0.3,
                    1.00
                ),
                "bagging_fraction": trial.suggest_float(
                    "bagging_fraction",
                    0.65,
                    1.00
                ),
                "bagging_freq": trial.suggest_int(
                    "bagging_freq",
                    1,
                    15
                ),
                "lambda_l1": trial.suggest_float(
                    "lambda_l1",
                    1e-5,
                    10.0,
                    log=True
                ),
                "lambda_l2": trial.suggest_float(
                    "lambda_l2",
                    1e-5,
                    10.0,
                    log=True
                )
            }

            min_required_depth = int(math.log2(params["num_leaves"])) + 1
            params["max_depth"] = max(params["max_depth"], min_required_depth)

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
                tags=["lgbm", level],
                reinit="finish_previous",
                dir="../../artifacts"
            )

            trainer = LGBMCVTrainer(
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
                    f"[ERROR] study={study_name} tr={trial.number}"
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
