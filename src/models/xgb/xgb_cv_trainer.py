from __future__ import annotations
import gc
import os
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
from sklearn.metrics import roc_auc_score

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
                gdf = gdf[~gdf[self.fold_col] != self.exclude_folds]

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
    """
    XGBを使ったCVトレーナー。

    Attributes
    ----------
    data_id: str
        使用するdata
    params : dict
        XGBのパラメータ。
    early_stopping_rounds : int, default 100
        早期停止ラウンド数。
    num_boost_round : int, default 20000
        iterationの最大値。
    seed : int, default 42
        乱数シード。
    """
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
            "colsample_bytree": 0.8,
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
            meta = {"row_id"}
            if self.target in all_cols:
                meta.add(self.target)
            if self.weight_col in all_cols:
                meta.add(self.weight_col)
            if self.fold_col:
                meta.add(self.fold_col)

            self.features = [
                c for c in all_cols
                if c not in meta and "fold" not in c
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
        loggers: list[CVLogger] | None = None
    ):
        """
        CVを用いてモデルを学習し、OOF予測とtest_dfの平均予測を返す。

        Returns
        -------
        oof_pred : ndarray
            OOF予測配列
        test_pred : ndarray
            test_dfに対する予測配列
        """
        if self.test_paths is None:
            raise ValueError("Please provide test_paths (got None).")

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

        iteration_list = []
        auc_scores = []
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
            test_it = ParquetIter(
                paths=self.test_paths,
                features=self.features,
                target=self.target,
                cat_cols=self.cat_cols,
                predict_mode=True,
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
            dtest = xgb.QuantileDMatrix(
                test_it,
                enable_categorical=True,
                ref=dtrain
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

            oof[val_idx] = model.predict(
                dvalid, iteration_range=(0, model.best_iteration + 1)
            )
            test_pred += model.predict(
                dtest,
                iteration_range=(0, model.best_iteration + 1)
            )

            t_fit_end = now()
            t_fold_end = now()

            print_duration(t_qdm_start, t_qdm_end, "\nQuantileDMatrix Build Time")
            print_duration(t_fit_start, t_fit_end)
            runtime = print_duration(
                t_fold_start, t_fold_end, f"Fold {i+1} Runtime"
            )

            best_iteration = model.best_iteration
            auc_train = evals_result["train"]["auc"][best_iteration]
            auc_valid = evals_result["valid"]["auc"][best_iteration]

            print(f"\nAUC Train: {auc_train:.5f}")
            print(f"AUC Valid: {auc_valid:.5f}\n")

            iteration_list.append(best_iteration)
            auc_scores.append(auc_valid)

            importances = model.get_score(importance_type="total_gain")
            total_gain = float(sum(importances.values()))
            df = pl.DataFrame(
                {
                    "Feature": list(importances.keys()),
                    "ImportanceRatio": [
                        ((v/total_gain)*100.0)/self.n_folds
                        for v in importances.values()
                    ],
                }
            )
            fi_fold_frames.append(df)

            fold_summary = {
                "auc": auc_valid,
                "best_iter": best_iteration,
                "runtime": runtime
            }

            for lg in loggers:
                lg.on_fold_end(
                    i,
                    "iter",
                    evals_result,
                    summary=fold_summary
                )

            del train_it, valid_it, test_it, dtrain, dvalid, dtest, model, val_idx
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            self.pmp.free_all_blocks()

        y = (
            pl.read_parquet(self.train_paths, columns=self.target)
              .get_column(self.target)
              .cast(pl.Float32)
              .to_numpy()
        )

        test_pred /= self.n_folds

        all_fi = pl.concat(fi_fold_frames, how="vertical_relaxed")
        fi_mean = (
            all_fi
            .group_by("Feature")
            .agg([
                pl.sum("ImportanceRatio").alias("mean_ratio")
            ])
        ).sort("mean_ratio", descending=True)

        auc_oof = roc_auc_score(y, oof)
        auc_mean = np.mean(auc_scores)
        auc_std = np.std(auc_scores)

        print(f"\n{' CV Results ':*^48}")
        print("─" * 48)
        print(f" {'Metric':^9}  {'OOF':>10}  {'Mean':>10} ± {'Std':<10} ")
        print("-" * 48)
        print(f" {'AUC':^9} "
              f" {auc_oof:>10.5f} "
              f" {auc_mean:>10.5f} ± {auc_std:<10.5f} ")
        print("─" * 48)

        print(f"Avg best iteration: {np.mean(iteration_list)}")

        t_total_end = now()
        total_runtime = print_duration(
            t_total_start, t_total_end, "Total CV Runtime"
        )
        print(f"Free CPU Mem: {round(free_ram_gib(), 2)} GB")
        print(f"Free GPU Mem: {round(free_vram_gib(), 2)} GB")

        result = {
            "oof": oof,
            "test_pred": test_pred,
            "oof_score": auc_oof,
            "fi_mean": fi_mean
        }
        overall_summary = {
            "auc_oof": auc_oof,
            "auc_mean": auc_mean,
            "auc_std": auc_std,
            "iter_mean": np.mean(iteration_list),
            "total_runtime": total_runtime
        }

        for lg in loggers:
            lg.on_end(overall_summary)

        return result

    def full_train(
        self,
        loggers: list[CVLogger] | None = None
    ):
        """
        訓練データ全体でモデルを学習し、test_dfに対する予測結果をnpy形式で返す

        Parameters
        ----------
        iterations : int
            学習の繰り返し回数。

        Returns
        -------
        test_prads : np.ndarray
            test dataの予測値
        """
        t_total_start = now()

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

        dtrain = xgb.QuantileDMatrix(train_it, enable_categorical=True)
        dtest = xgb.QuantileDMatrix(test_it, enable_categorical=True, ref=dtrain)
        t_qdm_end = now()
        t_fit_start = now()

        model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=[]
        )
        t_fit_end = now()

        print_duration(t_qdm_start, t_qdm_end, "\nQuantileDMatrix Build Time")
        print_duration(t_fit_start, t_fit_end)

        test_pred = model.predict(dtest)

        t_total_end = now()
        print_duration(t_total_start, t_total_end, "Total CV Runtime")

        return test_pred

    def fit_one_fold(
        self,
        fold_idx=0,
        loggers=None
    ):
        """
        指定した1つのfoldのみを用いてモデルを学習する。
        主にOptunaによるハイパーパラメータ探索時に使用。

        Parameters
        ----------
        fold : int
            学習に使うfold番号。

        Rerurn
        ------
        score : float
            Score
        """
        t_total_start = now()
        print(f"Free CPU Mem: {round(free_ram_gib(), 2)} GB")
        print(f"Free GPU Mem: {round(free_vram_gib(), 2)} GB")

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

        t_qdm_end = now()
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
        print_duration(t_qdm_start, t_qdm_end, "\nQuantileDMatrix Build Time")
        print_duration(t_fit_start, t_fit_end)

        best_iteration = model.best_iteration

        auc_train = evals_result["train"]["auc"][best_iteration]
        auc_valid = evals_result["valid"]["auc"][best_iteration]

        print(f"\nAUC Train: {auc_train:.5f}")
        print(f"AUC Valid: {auc_valid:.5f}")

        t_total_end = now()
        runtime = print_duration(t_total_start, t_total_end, "Total Runtime")

        fold_summary = {
            "auc": auc_valid,
            "best_iter": best_iteration,
            "runtime": runtime
        }
        for lg in loggers:
            lg.on_fold_end(
                fold_idx,
                "iter",
                evals_result,
                summary=fold_summary
            )

        del train_it, valid_it, dtrain, dvalid, model
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        self.pmp.free_all_blocks()

        print(f"Free CPU Mem: {round(free_ram_gib(), 2)} GB")
        print(f"Free GPU Mem: {round(free_vram_gib(), 2)} GB")

        return auc_valid
