from __future__ import annotations
import gc
import os
import re
import math
from dataclasses import dataclass, field
from time import perf_counter as now
from typing import Any, Iterable, Optional, List

import cudf
import rmm
import cupy as cp
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import rmm.mr as mr
from rmm.allocators.cupy import rmm_cupy_allocator
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

from src.utils.loggers import CVLogger, NoOpLogger
from src.utils.print_duration import print_duration
from src.utils.mem_info import free_ram_gib, free_vram_gib


@dataclass(eq=False)
class ParquetIter(xgb.core.DataIter):
    # === 引数（元 __init__ のシグネチャ） ===
    paths: list[str]

    features: list[str] = None
    target: str = "target"
    cat_cols: Optional[Iterable[str]] = None
    fold_col: Optional[str] = None
    include_fold: str = None
    exclude_fold: str = None
    weight_col: Optional[str] = None

    rowgroup_batch: int = 1
    gpu: Optional[bool] = None
    predict_mode: bool = False

    # === 内部状態（initの引数にしないもの） ===
    _temporary_data: Any = field(init=False, default=None, repr=False)
    _pass_count: int = field(init=False, default=0, repr=False)

    _files: List[dict] = field(init=False, default_factory=list, repr=False)
    _file_idx: int = field(init=False, default=0, repr=False)
    _rg_idx: int = field(init=False, default=0, repr=False)
    _columns: List[str] = field(init=False, default_factory=list, repr=False)

    def __post_init__(self) -> None:
        super().__init__()
        hdr = pl.read_parquet(self.paths, n_rows=0)
        all_cols = hdr.columns

        # 入力列（重複除去）
        cols = list(self.features)
        if (not self.predict_mode) and (self.target in all_cols):
            cols.append(self.target)
        if (not self.predict_mode) and (self.weight_col in all_cols):
            cols.append(self.weight_col)
        if (not self.predict_mode) and (self.fold_col in all_cols):
            cols.append(self.fold_col)
        self._columns = list(dict.fromkeys(cols))

        # 内部状態
        self._reader = None
        self._current_file_index = 0

        # 各ファイルの row group 数を先に調べておく
        self._files = []
        for p in self.paths:
            pf = pq.ParquetFile(p)
            self._files.append({"path": p, "nrg": pf.num_row_groups})

        self._file_idx = 0
        self._rg_idx = 0

    def reset(self):
        self._file_idx = 0
        self._rg_idx = 0
        self._pass_count += 1

    def next(self, input_data):
        """
        1 回呼ばれるごとに row group の束 (rowgroup_batch) を 1 塊だけ返す。
        """
        while True:
            if self._file_idx >= len(self._files):
                return 0  # 終了

            rec = self._files[self._file_idx]
            path, nrg = rec["path"], rec["nrg"]

            if self._rg_idx >= nrg:
                # 次のファイルへ
                self._file_idx += 1
                self._rg_idx = 0
                continue

            # このバッチで読む row groups
            start = self._rg_idx
            end = min(self._rg_idx + self.rowgroup_batch, nrg)
            bundle = list(range(start, end))
            self._rg_idx = end  # 次に備える

            # === ここがコア：cuDF で row group を直接読む ===
            # ※ cudf.read_parquet は単一ファイル向け。複数パスは「ループで回す」方針。
            gdf = cudf.read_parquet(path, columns=self._columns, row_groups=bundle)

            if self.include_fold is not None:
                gdf = gdf[gdf[self.fold_col] == self.include_fold]
            if self.exclude_fold is not None:
                gdf = gdf[gdf[self.fold_col] != self.exclude_fold]

            # カテゴリ化（必要な列のみ）
            if self.cat_cols:
                for c in self.cat_cols:
                    gdf[c] = gdf[c].astype("category")

            # 出力
            if self.predict_mode:
                input_data(data=gdf[self.features])
            else:
                kwargs = dict(data=gdf[self.features], label=gdf[self.target])
                if self.weight_col and self.weight_col in gdf.columns:
                    kwargs["weight"] = gdf[self.weight_col]
                input_data(**kwargs)

            # 後始末（参照を断つ→GC）
            del gdf
            gc.collect()
            return 1


@dataclass
class XGBCVTrainer:
    data_id: str
    train_paths: str | list[str]
    test_paths: str | list[str] | None = None

    features: Optional[list[str]] = None

    target: str = "target"
    fold_col: Optional[str] = None
    weight_col: Optional[str] = None
    cat_cols: Optional[list[str]] = None

    params: dict = field(default_factory=dict)

    n_folds: int = 5
    seed: int = 42
    gpu: bool = True

    opts: dict = field(init=True, default_factory=dict)

    def __post_init__(self):
        if isinstance(self.train_paths, (str, os.PathLike)):
            self.train_paths = [str(self.train_paths)]
        else:
            self.train_paths = [str(p) for p in self.train_paths]

        if self.test_paths:
            if isinstance(self.test_paths, (str, os.PathLike)):
                self.test_paths = [str(self.test_paths)]
            else:
                self.test_paths = [str(p) for p in self.test_paths]

        default_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "learning_rate": 0.1,
            "max_depth": 7,
            "min_child_weight": 10.0,
            "gamma": 0,
            "colsample_bytree": 0.4,
            "subsample": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "verbosity": 0,
            "tree_method": "hist",
            "device": "cuda",
            "seed": self.seed,
            "max_bin": 256,
            "grow_policy": "depthwise",
            "predictor": "gpu_predictor"
        }

        user_params = self.params or {}
        merged = {**default_params, **user_params}

        self.early_stopping_rounds = self.opts.get(
            "early_stopping_rounds",
            None
        )
        self.num_boost_round = self.opts.get(
            "num_boost_round",
            20000
        )

        # ユーザー未指定なら lr に応じて自動設定（下限あり）
        if self.early_stopping_rounds is None:
            lr = float(merged["learning_rate"])
            self.early_stopping_rounds = max(50, int(math.ceil(10.0 / lr)))

        # train() の引数として取り出す
        self.params = merged

        self.rep_metric = "auc"
        self.metrics = {
            "rmse": lambda y, p: np.sqrt(mse(y, p)),
            "r2": r2_score,
            "mae": lambda y, p: np.mean(np.abs(y-p)),
            # "mape": lambda y, p: np.mean(np.abs((y-p)/y)),
            "accuracy": lambda y, p: np.mean(
                [y_i == (1 if p_i > 0.5 else 0) for y_i, p_i in zip(y, p)]
            ),
            "log_loss": log_loss,
            "auc": roc_auc_score
        }

        hdr = pl.read_parquet(self.train_paths, n_rows=0)
        all_cols = hdr.columns

        if self.fold_col is None:
            self.fold_col = f"{self.n_folds}fold-s{self.seed}"

        if self.cat_cols is None:
            self.cat_cols = [
                c for c, dt in zip(hdr.columns, hdr.dtypes)
                if dt == pl.Categorical
            ]

        if self.fold_col not in all_cols:
            raise ValueError(f"fold_col not found in dataset: {self.fold_col}")
        else:
            print(f"Fold Col: {self.fold_col}")

        if self.features is None:
            meta = {
                c
                for c in ("row_id", self.target, self.weight_col, self.fold_col)
                if c and c in all_cols
            }
            pat = re.compile(r"^\d+fold(?:-[A-Za-z0-9]+)?$")
            self.features = [
                c for c in all_cols
                if c not in meta and not pat.fullmatch(c)
            ]

        dev_mr = mr.CudaAsyncMemoryResource()
        mr.set_current_device_resource(dev_mr)
        rmm.reinitialize(
            managed_memory=False,
            initial_pool_size=None,
        )
        cp.cuda.set_allocator(rmm_cupy_allocator)

        cp.get_default_memory_pool().set_limit(4 * 1024**3)
        self.pmp = cp.cuda.PinnedMemoryPool()
        cp.cuda.set_pinned_memory_allocator(self.pmp.malloc)

    def fit(
        self,
        loggers: list[CVLogger] | None = None,
        one_fold: bool = False
    ) -> dict:
        t_total_start = now()

        loggers = loggers or [NoOpLogger()]
        meta = {
            "data_id": self.data_id,
            "seed": self.seed,
            "n_folds": self.n_folds,
            "early_stopping_rounds": self.early_stopping_rounds,
            **self.params
        }
        for lg in loggers:
            lg.on_start(meta)

        if not one_fold:
            train_rows = (
                pl.scan_parquet(self.train_paths)
                  .select(pl.len())
                  .collect()
                  .item()
            )
            test_rows = (
                pl.scan_parquet(self.test_paths)
                  .select(pl.len())
                  .collect()
                  .item()
            )

            oof = np.zeros(train_rows, dtype=np.float32)
            test_pred = np.zeros(test_rows, dtype=np.float32)

            test_it = ParquetIter(
                paths=self.test_paths,
                features=self.features,
                target=self.target,
                cat_cols=self.cat_cols,
                predict_mode=True,
                gpu=self.gpu
            )

        iteration_list = []
        fold_scores = {name: [] for name in self.metrics.keys()}
        fi_fold_frames = []

        for i in range(self.n_folds):
            title = f" Fold {i + 1} / {self.n_folds} "
            print("=" * 48)
            print(f"{title:=^48}")
            print("=" * 48)
            print(f"Free CPU Mem: {round(free_ram_gib(), 2)} GB")
            print(f"Free GPU Mem: {round(free_vram_gib(), 2)} GB")

            t_fold_start = now()
            t_qdm_start = now()
            fold_summary = {}

            train_it = ParquetIter(
                paths=self.train_paths,
                features=self.features,
                target=self.target,
                cat_cols=self.cat_cols,
                fold_col=self.fold_col,
                exclude_fold=i,
                weight_col=self.weight_col,
                gpu=True
            )
            valid_it = ParquetIter(
                paths=self.train_paths,
                features=self.features,
                target=self.target,
                cat_cols=self.cat_cols,
                fold_col=self.fold_col,
                include_fold=i,
                weight_col=self.weight_col,
                gpu=self.gpu
            )

            dtrain = xgb.QuantileDMatrix(
                train_it,
                enable_categorical=True
            )
            dvalid = xgb.QuantileDMatrix(
                valid_it,
                enable_categorical=True,
                ref=dtrain
            )

            if not one_fold:
                dtest = xgb.QuantileDMatrix(
                    test_it,
                    enable_categorical=True,
                    ref=dtrain
                )

            y_valid = (
                pl.read_parquet(
                    self.train_paths, columns=[self.target, self.fold_col]
                ).filter(pl.col(self.fold_col) == i)
                .select(self.target)
                .to_numpy()
                .astype(np.int32)
                .ravel()
            )

            val_idx = (
                pl.scan_parquet(self.train_paths)
                  .select(["row_id", self.fold_col])
                  .filter(pl.col(self.fold_col) == i)
                  .select("row_id")
                  .collect()
                  .get_column("row_id")
                  .to_numpy()
                  .astype(np.int32, copy=False)
            )

            evals_result = {}

            t_qdm_end = now()
            print_duration(t_qdm_start, t_qdm_end, "\nQuantileDMatrix Build Time")

            t_fit_start = now()

            model = xgb.train(
                self.params,
                dtrain,
                num_boost_round=self.num_boost_round,
                evals=[(dtrain, "train"), (dvalid, "valid")],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=100,
                evals_result=evals_result,
            )

            val_pred = model.predict(
                dvalid, iteration_range=(0, model.best_iteration + 1)
            )
            if not one_fold:
                oof[val_idx] = val_pred
                test_pred += model.predict(
                    dtest,
                    iteration_range=(0, model.best_iteration + 1)
                )

            t_fit_end = now()

            print_duration(t_fit_start, t_fit_end)

            best_iter = model.best_iteration
            fold_summary["best_iter"] = best_iter

            for name, metric_func in self.metrics.items():
                val_score = metric_func(y_valid, val_pred)
                print(f"{name.upper()} Valid: {val_score:.5f}")
                fold_summary[name] = val_score
                fold_scores[name].append(val_score)

            iteration_list.append(best_iter)

            importances = model.get_score(importance_type="total_gain")
            total_gain = float(sum(importances.values()))
            df = pl.DataFrame(
                {
                    "Feature": list(importances.keys()),
                    "Importance": [
                        ((v/total_gain)*100.0)/self.n_folds
                        for v in importances.values()
                    ],
                }
            )
            fi_fold_frames.append(df)

            t_fold_end = now()
            fold_summary["runtime"] = print_duration(
                t_fold_start, t_fold_end, f"\nFold {i+1} Runtime"
            )

            for lg in loggers:
                lg.on_fold_end(
                    i,
                    "iter",
                    evals_result,
                    summary=fold_summary
                )

            if one_fold:
                result = {
                    "oof": None,
                    "test_pred": None,
                    "oof_score": fold_scores[self.rep_metric][0],
                    "fi_mean": df
                }
            else:
                del dtest

            del model, train_it, valid_it, dtrain, dvalid, val_idx
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            self.pmp.free_all_blocks()

            if one_fold:
                return result

        y = (
            pl.read_parquet(self.train_paths, columns=self.target)
              .get_column(self.target)
              .cast(pl.Float32)
              .to_numpy()
        )
        test_pred /= self.n_folds

        oofs = {
            name: metric_func(y, oof)
            for name, metric_func in self.metrics.items()
        }

        oof_stats = {
            name: {
                "oof": oofs[name],
                "mean": np.mean(vals),
                "std": np.std(vals)
            }
            for name, vals in fold_scores.items()
        }

        all_fi = pl.concat(fi_fold_frames, how="vertical_relaxed")
        fi_mean = (
            all_fi
            .group_by("Feature")
            .agg([
                pl.sum("Importance").alias("Importance")
            ])
        ).sort("Importance", descending=True)

        iter_mean = np.mean(iteration_list)

        print(f"\n{' CV Results ':*^48}")
        print("─" * 48)
        print(f" {'Metric':^9}  {'OOF':>10}  {'Mean':>10} ± {'Std':<10} ")
        print("-" * 48)
        for name, stats in oof_stats.items():
            print(f" {name.upper():^9} "
                  f" {stats['oof']:>10.5f} "
                  f" {stats['mean']:>10.5f} ± {stats['std']:<10.5f} ")
        print("─" * 48)
        print(f"Avg best iteration: {iter_mean}")

        print(f"Free CPU Mem: {round(free_ram_gib(), 2)} GB")
        print(f"Free GPU Mem: {round(free_vram_gib(), 2)} GB")

        result = {
            "oof": oof,
            "test_pred": test_pred,
            "oof_score": oofs[self.rep_metric],
            "fi_mean": fi_mean
        }
        overall_summary = {"iter_mean": iter_mean}
        for name, stats in oof_stats.items():
            overall_summary[f"{name}_mean"] = stats["mean"]
            overall_summary[f"{name}_std"] = stats["std"]
            overall_summary[f"{name}_oof"] = oofs[name]

        t_total_end = now()
        overall_summary["total_runtime"] = print_duration(
            t_total_start, t_total_end, "Total CV Runtime"
        )

        for lg in loggers:
            lg.on_end(overall_summary)

        return result

    def full_train(
        self,
        loggers: list[CVLogger] | None = None
    ) -> dict:
        t_total_start = now()

        loggers = loggers or [NoOpLogger()]
        meta = {
            "data_id": self.data_id,
            "seed": self.seed,
            "n_folds": self.n_folds,
            "early_stopping_rounds": self.early_stopping_rounds,
            **self.params
        }
        for lg in loggers:
            lg.on_start(meta)

        test_rows = pq.ParquetFile(self.test_path).metadata.num_rows
        test_pred = np.zeros(test_rows, dtype=np.float32)

        self.num_boost_round = (
            self.num_boost_round * (self.n_folds/(self.n_folds-1))
        )

        t_qdm_start = now()

        train_it = ParquetIter(
            paths=self.train_paths,
            features=self.features,
            target=self.target,
            cat_cols=self.cat_cols,
            fold_col=self.fold_col,
            weight_col=self.weight_col,
            gpu=self.gpu
        )

        test_it = ParquetIter(
            paths=self.test_paths,
            features=self.features,
            target=self.target,
            cat_cols=self.cat_cols,
            weight_col=self.weight_col,
            predict_mode=True,
            gpu=self.gpu
        )

        dtrain = xgb.QuantileDMatrix(
            train_it,
            enable_categorical=True
        )
        dtest = xgb.QuantileDMatrix(
            test_it,
            enable_categorical=True,
            ref=dtrain
        )
        t_qdm_end = now()
        print_duration(t_qdm_start, t_qdm_end, "\nQuantileDMatrix Build Time")

        t_fit_start = now()

        model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=[]
        )
        t_fit_end = now()

        print_duration(t_fit_start, t_fit_end)

        test_pred = model.predict(dtest)

        t_total_end = now()
        total_runtime = print_duration(
            t_total_start, t_total_end, "Total Runtime"
        )
        print(f"Free CPU Mem: {round(free_ram_gib(), 2)} GB")
        print(f"Free GPU Mem: {round(free_vram_gib(), 2)} GB")

        result = {
            "oof": None,
            "test_pred": test_pred,
            "oof_score": None
        }
        overall_summary = {
            "total_runtime": total_runtime
        }
        for lg in loggers:
            lg.on_end(overall_summary)

        return result

    def fit_one_fold(
        self,
        fold_idx: int = 0,
        loggers: bool = None
    ) -> float:
        t_total_start = now()
        fold_summary = {}

        loggers = loggers or [NoOpLogger()]
        meta = {
            "data_id": self.data_id,
            "seed": self.seed,
            "n_folds": self.n_folds,
            "early_stopping_rounds": self.early_stopping_rounds,
            **self.params
        }
        for lg in loggers:
            lg.on_start(meta)

        print(f"Free CPU Mem: {round(free_ram_gib(), 2)} GB")
        print(f"Free GPU Mem: {round(free_vram_gib(), 2)} GB")

        evals_result = {}

        t_qdm_start = now()

        train_it = ParquetIter(
            paths=self.train_paths,
            features=self.features,
            target=self.target,
            cat_cols=self.cat_cols,
            fold_col=self.fold_col,
            exclude_fold=fold_idx,
            weight_col=self.weight_col,
            gpu=self.gpu
        )
        valid_it = ParquetIter(
            paths=self.train_paths,
            features=self.features,
            target=self.target,
            cat_cols=self.cat_cols,
            fold_col=self.fold_col,
            include_fold=fold_idx,
            weight_col=self.weight_col,
            gpu=self.gpu
        )

        dtrain = xgb.QuantileDMatrix(
            train_it,
            enable_categorical=True
        )
        dvalid = xgb.QuantileDMatrix(
            valid_it,
            enable_categorical=True,
            ref=dtrain
        )

        y_valid = (
            pl.read_parquet(
                self.train_paths, columns=[self.target, self.fold_col]
            ).filter(pl.col(self.fold_col) == fold_idx)
            .select(self.target)
            .to_numpy()
            .astype(np.int32)
            .ravel()
        )

        t_qdm_end = now()
        print_duration(t_qdm_start, t_qdm_end, "\nQuantileDMatrix Build Time")

        t_fit_start = now()

        model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=100,
            evals_result=evals_result,
        )

        t_fit_end = now()
        print_duration(t_fit_start, t_fit_end)

        fold_summary["best_iter"] = model.best_iteration

        val_pred = model.predict(
            dvalid, iteration_range=(0, model.best_iteration + 1)
        )
        for name, metric_func in self.metrics.items():
            val_score = metric_func(y_valid, val_pred)
            print(f"{name.upper()} Valid: {val_score:.5f}")
            fold_summary[name] = val_score

        del train_it, valid_it, dtrain, dvalid, y_valid, model
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        self.pmp.free_all_blocks()

        t_total_end = now()
        fold_summary["runtime"] = print_duration(
            t_total_start, t_total_end, "Total CV Runtime"
        )
        print(f"Free CPU Mem: {round(free_ram_gib(), 2)} GB")
        print(f"Free GPU Mem: {round(free_vram_gib(), 2)} GB")

        for lg in loggers:
            lg.on_fold_end(
                fold_idx,
                "iter",
                evals_result,
                summary=fold_summary
            )

        return fold_summary[self.rep_metric]
