from __future__ import annotations
import os
from typing import Callable, Any
import json
from pathlib import Path

import optuna
import wandb
from optuna.exceptions import TrialPruned

from src.models.model_registry import get_trainer
from src.utils.snapshot_study import snapshot_study
from src.utils.telegram import send_message
from src.utils.loggers import WandbLogger


# ---- Types ----
Objective = Callable[[optuna.trial.Trial], float]
ObjectiveFactory = Callable[..., Objective]
SearchSpace = Callable[[optuna.trial.Trial], dict[str, Any]]


def space_xgb(t: optuna.trial.Trial) -> dict[str, Any]:
    return {
        "learning_rate": t.suggest_float("learning_rate", 0.1, 0.1),
        "max_depth": t.suggest_int("max_depth", 6, 13),
        "min_child_weight": t.suggest_float("min_child_weight", 0.0, 100.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.3, 0.7),
        "subsample": t.suggest_float("subsample", 0.5, 0.9),
        "reg_alpha": t.suggest_float("reg_alpha", 1e-4, 40.0, log=True),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }


def space_lgbm(t: optuna.trial.Trial) -> dict[str, Any]:
    return {
        "learning_rate": t.suggest_float("learning_rate", 0.02, 0.02),
        "num_leaves": t.suggest_int("num_leaves", 300, 1200),
        "min_child_samples": t.suggest_int("min_child_samples", 100, 20000),
        "min_split_gain": t.suggest_float("min_split_gain", 1e-5, 10, log=True),
        "feature_fraction": t.suggest_float("feature_fraction", 0.3, 0.5),
        "bagging_fraction": t.suggest_float("bagging_fraction", 0.80, 0.95),
        "bagging_freq": t.suggest_int("bagging_freq", 1, 15),
        "lambda_l1": t.suggest_float("lambda_l1", 1e-5, 10.0, log=True),
        "lambda_l2": t.suggest_float("lambda_l2", 1e-5, 10.0, log=True)
    }


def space_cb(t: optuna.trial.Trial) -> dict[str, Any]:
    return {
        "learning_rate": t.suggest_float("learning_rate", 0.1, 0.1),
        "depth": t.suggest_int("depth", 6, 16),
        "min_data_in_leaf": t.suggest_int("min_data_in_leaf", 1, 100),
        "random_strength": t.suggest_float("random_strength", 1, 80),
        "l2_leaf_reg": t.suggest_int("l2_leaf_reg", 1, 10),
        "bagging_temperature": t.suggest_float("bagging_temperature", 1e-2, 10)
    }


def space_rfr(t: optuna.trial.Trial) -> dict[str, Any]:
    return {
        "n_estimators": t.suggest_int("n_estimators", 50, 150),
        "max_depth": t.suggest_int("max_depth", 4, 30)
    }


def space_rfc(t: optuna.trial.Trial) -> dict[str, Any]:
    return {
        "n_estimators": t.suggest_int("n_estimators", 50, 150),
        "max_depth": t.suggest_int("max_depth", 4, 30)
    }


def space_mlp(t: optuna.trial.Trial) -> dict[str, Any]:
    num_layers = t.suggest_int("num_layers", 1, 4)

    hidden_dim1 = t.suggest_int("hidden_dim1", 256, 1024, step=32)

    if num_layers >= 2:
        hidden_dim2 = t.suggest_int("hidden_dim2", 128, hidden_dim1, step=32)
    else:
        hidden_dim2 = -1
        t.suggest_int("hidden_dim2", -1, -1)

    if num_layers >= 3:
        hidden_dim3 = t.suggest_int("hidden_dim3", 64, hidden_dim2, step=32)
    else:
        hidden_dim3 = -1
        t.suggest_int("hidden_dim3", -1, -1)

    if num_layers >= 4:
        hidden_dim4 = t.suggest_int("hidden_dim4", 32, hidden_dim3, step=32)
    else:
        hidden_dim4 = -1
        t.suggest_int("hidden_dim4", -1, -1)

    return {
        "hidden_dim1": hidden_dim1,
        "hidden_dim2": hidden_dim2,
        "hidden_dim3": hidden_dim3,
        "hidden_dim4": hidden_dim4,
        "batch_size": t.suggest_int("batch_size", 512, 1120, step=32),
        "lr": t.suggest_float("lr", 1e-3, 1e-1, log=True),
        "eta_min": t.suggest_float("eta_min", 1e-4, 1e-3, log=True),
        "dropout_rate": round(t.suggest_float(
            "dropout_rate", 0.1, 0.6, step=0.05), 2),
        "activation": t.suggest_categorical(
            "activation", [
                "ReLU",
                "LeakyReLU",
                "GELU",
                "SiLU"
            ]
        ),
    }


def space_realmlp(t: optuna.trial.Trial) -> dict[str, Any]:
    return {}


def space_tabnet(t: optuna.trial.Trial) -> dict[str, Any]:
    return {
        "n_d": t.suggest_int("n_d", 8, 24),
        "n_a": t.suggest_int("n_a", 8, 24),
        "n_steps": t.suggest_int("n_steps", 1, 10),
        "gamma": t.suggest_float("gamma", 1.2, 2.0),
        "n_independent": t.suggest_int("n_independent", 1, 4),
        "n_shared": t.suggest_int("n_shared", 1, 4),
        "momentum": t.suggest_float("momentum", 0.02, 0.4),
        "lambda_sparse": t.suggest_float("lambda_sparse", 1e-5, 1e-3, log=True),
        "lr": t.suggest_float("lr", 1e-4, 1e-3),
        "batch_size": t.suggest_int("batch_size", 5240, 10480, step=32),
        "eta_min": t.suggest_float("eta_min", 1e-4, 1e-3, log=True),
        "mask_type": t.suggest_categorical("mask_type", ["entmax", "sparsemax"])
    }


def space_logreg(t: optuna.trial.Trial) -> dict[str, Any]:
    return {
        "C": t.suggest_float("C", 1e-2, 1e2, log=True)
    }


def space_lasso(t: optuna.trial.Trial) -> dict[str, Any]:
    return {
        "alpha": t.suggest_float("alpha", 1e-2, 1e2, log=True)
    }


def space_ridge(t: optuna.trial.Trial) -> dict[str, Any]:
    return {
        "alpha": t.suggest_float("alpha", 1e-2, 1e2, log=True)
    }


_REGISTRY: dict[str, SearchSpace] = {
    "xgb": space_xgb,
    "lgbm": space_lgbm,
    "cb": space_cb,
    "rfr": space_rfr,
    "rfc": space_rfc,
    "mlp": space_mlp,
    "realmlp": space_realmlp,
    "tabnet": space_tabnet,
    "logreg": space_logreg,
    "lasso": space_lasso,
    "ridge": space_ridge,

}


def create_objective(
    model_type: str,
    data_id: int,
    seed: int = 42,
    n_folds: int = 5,
    wandb_project: str = "project",
    study_name: str = "study",
    opts: dict | None = None
):
    Trainer = get_trainer(model_type)
    space_fn = _REGISTRY[model_type]

    optuna_dir = Path("../../artifacts/optuna")
    os.makedirs(optuna_dir / f"{study_name}", exist_ok=True)

    with open(f"../../artifacts/features/{data_id}/meta.json")as f:
        m = json.load(f)

    train_paths = m["train_paths"]
    level = m["level"]

    def objective(trial):
        params = space_fn(trial)
        run = wandb.init(
            project=wandb_project,
            group=study_name,
            name=f"trl{trial.number}",
            job_type="optuna-search",
            config={
                "data_id": data_id,
                "n_folds": n_folds,
                "level": level,
                "model": model_type,
            },
            tags=[model_type, level],
            reinit="finish_previous",
            dir="../../artifacts"
        )
        try:
            trainer = Trainer(
                data_id,
                train_paths,
                n_folds=n_folds,
                params=params,
                seed=seed,
                opts=opts
            )

            result = trainer.fit(
                loggers=[WandbLogger(run=run)],
                one_fold=True
            )

            path = optuna_dir / f"{study_name}/trl{trial.number}.json"
            manifest = {
                "params": params,
                "n_folds": n_folds,
                "seed": seed,
                "wandb_id": run.id,
                "wandb_url": run.url,
                "opts": opts,
                "score": float(result["oof_score"])
            }
            with open(path, "w") as f:
                json.dump(manifest, f, indent=4)

            return result["oof_score"]
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
