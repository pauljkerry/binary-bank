from dataclasses import dataclass
from typing import Protocol, Optional, Any

import numpy as np
import polars as pl


@dataclass
class CVResult:
    oof: Any
    test_pred: Optional[Any]
    oof_score: float
    mean: float
    std: float
    iteration_list: list[int]
    fi_mean: Optional[pl.DataFrame]


# ===== Logger Protocol =====
class CVLogger(Protocol):
    def on_start(self, meta: dict) -> None: ...

    def on_fold_end(
        self,
        fold_idx: int,
        fold_score: float,
        evals_result: dict,
        best_iteration: int
    ) -> None: ...
    def on_end(self, result: CVResult) -> None: ...


# ===== No-op Logger =====
class NoOpLogger:
    def on_start(self, meta: dict) -> None:
        pass

    def on_fold_end(
        self,
        fold_idx: int,
        fold_score: float,
        evals_result: dict,
        best_iteration: int
    ) -> None:
        pass

    def on_end(self, result: CVResult) -> None:
        pass


# ===== Weights & Biases Logger =====
class WandbLogger:
    def __init__(self, run=None, prefix: str = ""):
        import wandb
        self.wandb = wandb
        self.run = run or wandb.run or wandb.init()
        self.p = prefix.rstrip("/")  # "cv"など。空でもOK。
        self._metrics_defined = False

    def _k(self, name: str) -> str:
        """prefix付きのキー名を返す"""
        return f"{self.p}/{name}" if self.p else name

    def on_start(self, meta: dict) -> None:
        self.run.config.update(meta, allow_val_change=True)
        if not self._metrics_defined:
            self.wandb.define_metric("iter")                          # 独自X軸
            self.wandb.define_metric("train/*", step_metric="iter")
            self.wandb.define_metric("eval/*",  step_metric="iter")
            self._metrics_defined = True

    def on_fold_end(
        self,
        fold_idx: int,
        fold_score: float,
        evals_result: dict,
        best_iteration: int
    ) -> None:
        fno = fold_idx + 1
        self.wandb.log({self._k(f"auc_f{fno}"): fold_score})

        self.run.summary[self._k(f"best_iter_f{fno}")] = best_iteration

        # evals_result を時系列でlog（大きすぎるならスキップ間引き推奨）
        # 期待形：{"train": {"auc": [..], "logloss": [..]}, "eval": {...}}
        # どのsplitにもメトリクスがある前提で長さを取得
        try:
            first_split = next(iter(evals_result.values()))
            length = len(next(iter(first_split.values())))
        except StopIteration:
            length = 0

        train_auc = evals_result["train"]["auc"]
        valid_auc = evals_result["eval"]["auc"]
        for i in range(length):
            self.wandb.log({
                "iter": i,                                  # ← 自前のx軸
                f"train/auc_f{fno}": train_auc[i],   # y軸の値（スカラー）
                f"eval/auc_f{fno}":  valid_auc[i]
            })

    def on_end(self, result: CVResult) -> None:
        self.run.summary[self._k("auc_oof")] = result.oof_score
        self.run.summary[self._k("auc_mean")] = result.mean
        self.run.summary[self._k("auc_std")] = result.std
        self.run.summary[self._k("iter_mean")] = np.mean(result.iteration_list)
        self.wandb.finish()