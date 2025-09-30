from typing import Protocol


# ===== Logger Protocol =====
class CVLogger(Protocol):
    def on_start(self, meta: dict) -> None: ...

    def on_fold_end(
        self,
        fold_idx: int,
        axis_name: str | None = None,
        evals_result: dict | None = None,
        extra: dict | None = None,
        summary: dict | None = None
    ) -> None: ...

    def on_end(self, summary: dict) -> None: ...


# ===== No-op Logger =====
class NoOpLogger:
    def on_start(self, meta: dict) -> None:
        pass

    def on_fold_end(
        self,
        fold_idx: int,
        axis_name: str | None = None,
        evals_result: dict | None = None,
        extra: dict | None = None,
        summary: dict | None = None
    ) -> None:
        pass

    def on_end(self, summary: dict) -> None:
        pass


# ===== Weights & Biases Logger =====
class WandbLogger:
    def __init__(self, run=None, prefix: str = ""):
        import wandb
        self.wandb = wandb
        self.run = run or wandb.run or wandb.init()
        self.p = prefix.rstrip("/")
        self._defined_folds: set[int] = set()

    def _k(self, name: str) -> str:
        return f"{self.p}/{name}" if self.p else name

    def _define_fold_metrics(self, axis_name: str, fno: int):
        if fno in self._defined_folds:
            return
        # 例: "train/*_f1", "eval/*_f1", "meta/*_f1" は x 軸として "t_f1" を使う
        step_key = self._k(f"{axis_name}_f{fno}")
        self.wandb.define_metric(self._k(f"train/f{fno}/*"), step_metric=step_key)
        self.wandb.define_metric(self._k(f"valid/f{fno}/*"), step_metric=step_key)
        self.wandb.define_metric(self._k(f"meta/f{fno}/*"),  step_metric=step_key)
        self._defined_folds.add(fno)

    def on_start(self, meta: dict) -> None:
        self.run.config.update(meta, allow_val_change=True)

    def on_fold_end(
        self,
        fold_idx: int,
        axis_name: str | None = None,
        evals_result: dict | None = None,
        extra: dict | None = None,
        summary: dict | None = None
    ) -> None:
        fno = fold_idx + 1
        self._define_fold_metrics(axis_name, fno)

        if evals_result:
            lengths = [
                len(arr)
                for md in evals_result.values() for arr in md.values()
            ]
        if extra:
            lengths += [len(arr) for arr in extra.values()]
        if evals_result or extra:
            L = min(lengths) if lengths else 0

            for t in range(L):
                payload = {self._k(f"{axis_name}_f{fno}"): t}
                for split, md in evals_result.items():
                    for mname, arr in md.items():
                        payload[self._k(f"{split}/f{fno}/{mname}")] = float(arr[t])
                if extra:
                    for name, arr in extra.items():
                        payload[self._k(f"meta/f{fno}/{name}")] = float(arr[t])
                self.wandb.log(payload)

        if summary:
            for k, v in summary.items():
                self.run.summary[self._k(f"{k}_f{fno}")] = (
                    float(v) if isinstance(v, (int, float)) else v
                )

    def on_end(self, summary: dict) -> None:
        for k, v in summary.items():
            self.run.summary[self._k(k)] = (
                float(v)
                if isinstance(v, (int, float))
                else v
            )
        self.wandb.finish()