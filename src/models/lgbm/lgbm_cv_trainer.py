import gc
import math
import os
from dataclasses import dataclass, field
from time import perf_counter as now
from typing import Optional

import lightgbm as lgb
import numpy as np
import polars as pl

from sklearn.metrics import roc_auc_score

from src.utils.loggers import CVLogger, NoOpLogger
from src.utils.print_duration import print_duration
from src.utils.mem_info import free_ram_gib, free_vram_gib


@dataclass(eq=False)
class LGBMCVTrainer:
    """
    LGBMを使ったCVトレーナー。

    Attributes
    ----------
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

        self.lf_train = pl.scan_parquet(self.train_paths)
        self.lf_test = (
            pl.scan_parquet(self.test_paths) if self.test_paths else None
        )

        default_params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.1,
            "num_leaves": 500,
            "max_depth": -1,
            "min_child_samples": 100,
            "min_split_gain": 0,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "lambda_l1": 0.0,
            "lambda_l2": 0.0,
            "n_jobs": 25,
            "verbosity": -1,
            "random_state": self.seed
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

    def fit(
        self,
        loggers: list[CVLogger] | None = None
    ):
        """
        CVを用いてモデルを学習し、OOF予測とtest_dfの平均予測を返す。

        Returns
        -------
        oof_preds : ndarray
            OOF予測配列
        test_preds : ndarray
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

            need_cols = self.features + [self.target, "row_id"]
            train = (
                self.lf_train
                .filter(pl.col(self.fold_col) != i)
                .select(need_cols)
                .collect(engine="streaming")
            )
            valid = (
                self.lf_train
                .filter(pl.col(self.fold_col) == i)
                .select(need_cols)
                .collect(engine="streaming")
            )
            X_train = (
                train
                .select(self.features)
                .to_numpy()
                .astype(np.float32)
            )
            y_train = (
                train
                .select(self.target)
                .to_numpy()
                .astype(np.int32)
                .ravel()
            )
            X_valid = (
                valid
                .select(self.features)
                .to_numpy()
                .astype(np.float32)
            )
            y_valid = (
                valid
                .select(self.target)
                .to_numpy()
                .astype(np.int32)
                .ravel()
            )
            val_idx = (
                valid
                .select("row_id")
                .to_series()
                .to_numpy()
                .astype(np.int32, copy=False)
            )
            test = (
                self.lf_test
                .select(self.features)
                .collect(engine="streaming")
                .to_numpy()
                .astype(np.float32)
            )

            dtrain = lgb.Dataset(
                X_train,
                label=y_train,
                feature_name=self.features,
                categorical_feature=self.cat_cols,
            )

            dvalid = lgb.Dataset(
                X_valid,
                label=y_valid,
                feature_name=self.features,
                reference=dtrain
            )

            evals_result = {}

            model = lgb.train(
                self.params,
                dtrain,
                num_boost_round=self.num_boost_round,
                valid_sets=[dtrain, dvalid],
                valid_names=["train", "eval"],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=self.early_stopping_rounds),
                    lgb.record_evaluation(evals_result),
                    lgb.log_evaluation(period=100)
                ]
            )

            oof[val_idx] = model.predict(X_valid)
            test_pred += model.predict(test)

            best_iter = model.best_iteration
            train_score = evals_result["train"]["auc"][best_iter-1]
            eval_score = evals_result["eval"]["auc"][best_iter-1]
            print(f"Train AUC: {train_score:.5f}")
            print(f"Valid AUC: {eval_score:.5f}")

            auc_scores.append(eval_score)
            iteration_list.append(best_iter)

            importances = model.feature_importance(importance_type="gain")
            total_gain = importances.sum()
            df = pl.DataFrame(
                {
                    "Feature": model.feature_name(),
                    "ImportanceRatio": [
                        ((v/total_gain)*100.0)/self.n_folds for v in importances
                    ],
                }
            )
            fi_fold_frames.append(df)

            t_fold_end = now()
            runtime = print_duration(
                t_fold_start, t_fold_end, f"Fold {i+1} Runtime"
            )
            fold_summary = {
                "auc": eval_score,
                "runtime": runtime
            }

            for lg in loggers:
                lg.on_fold_end(
                    i,
                    axis_name="iter",
                    evals_result=evals_result,
                    summary=fold_summary
                )
            del model, X_train, y_train, X_valid, y_valid, test
            gc.collect()

        print("\n=== CV Results ===")
        y = (
            self.lf_train
            .select(self.target)
            .cast(pl.Float32)
            .collect(engine="streaming")
            .to_numpy()
            .ravel()
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

        oof_score = roc_auc_score(y, oof)

        auc_mean = np.mean(auc_scores)
        auc_std = np.std(auc_scores)
        iter_mean = np.mean(iteration_list)

        print(f"OOF score: {oof_score:.5f}")
        print(
            f"Mean: {auc_mean:.5f}, "
            f"Std: {auc_std:.5f}"
        )
        print(f"Avg best iteration: {iter_mean}")

        t_total_end = now()
        total_runtime = print_duration(
            t_total_start, t_total_end, "Total CV Runtime"
        )
        print(f"Free CPU Mem: {round(free_ram_gib(), 2)} GB")
        print(f"Free GPU Mem: {round(free_vram_gib(), 2)} GB")

        result = {
            "oof": oof,
            "test_pred": test_pred,
            "oof_score": oof_score,
            "fi_mean": fi_mean
        }
        overall_summary = {
            "auc_oof": oof_score,
            "auc_mean": auc_mean,
            "auc_std": auc_std,
            "iter_mean": iter_mean,
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
        訓練データ全体でモデルを学習し、test_dfに対する予測結果をnpy形式で保存する。

        Parameters
        ----------
        Returns
        -------
        test_preds : np.ndarray
            test dataの予測値
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

        test_rows = (
            pl.scan_parquet(self.test_paths)
              .select(pl.len())
              .collect()
              .item()
        )

        test_pred = np.zeros(test_rows, dtype=np.float32)

        need_cols = self.features + [self.target]
        train = (
            self.lf_train
            .select(need_cols)
            .collect(engine="streaming")
        )
        X_train = (
            train
            .select(self.features)
            .to_numpy()
            .astype(np.float32)
        )
        y_train = (
            train
            .select(self.target)
            .to_numpy()
            .astype(np.int32)
            .ravel()
        )
        test = (
            self.lf_test
            .select(self.features)
            .collect(engine="streaming")
            .to_numpy()
            .astype(np.float32)
        )

        dtrain = lgb.Dataset(
            X_train,
            label=y_train,
            feature_name=self.features,
            categorical_feature=self.cat_cols,
        )

        model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            valid_sets=[dtrain],
            valid_names=["train"]
        )

        test_pred += model.predict(test)

        test_pred /= self.n_folds

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
        fold_idx=0,
        loggers: list[CVLogger] | None = None
    ):
        """
        指定した1つのfoldのみを用いてモデルを学習する。
        主にOptunaによるハイパーパラメータ探索時に使用。

        Parameters
        ----------
        fold : int
            学習に使うfold番号。

        Return
        ------
        eval_score : float
            Score
        """
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

        print(f"Free CPU Mem: {round(free_ram_gib(), 2)} GB")
        print(f"Free GPU Mem: {round(free_vram_gib(), 2)} GB")

        need_cols = self.features + [self.target, "row_id"]
        train = (
            self.lf_train
            .filter(pl.col(self.fold_col) != fold_idx)
            .select(need_cols)
            .collect(engine="streaming")
        )
        valid = (
            self.lf_train
            .filter(pl.col(self.fold_col) == fold_idx)
            .select(need_cols)
            .collect(engine="streaming")
        )
        X_train = (
            train
            .select(self.features)
            .to_numpy()
            .astype(np.float32)
        )
        y_train = (
            train
            .select(self.target)
            .to_numpy()
            .astype(np.int32)
            .ravel()
        )
        X_valid = (
            valid
            .select(self.features)
            .to_numpy()
            .astype(np.float32)
        )
        y_valid = (
            valid
            .select(self.target)
            .to_numpy()
            .astype(np.int32)
            .ravel()
        )

        dtrain = lgb.Dataset(
            X_train,
            label=y_train,
            feature_name=self.features,
            categorical_feature=self.cat_cols,
        )

        dvalid = lgb.Dataset(
            X_valid,
            label=y_valid,
            feature_name=self.features,
            reference=dtrain
        )

        evals_result = {}

        model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.early_stopping_rounds),
                lgb.record_evaluation(evals_result),
                lgb.log_evaluation(period=100)
            ]
        )

        best_iter = model.best_iteration
        train_score = evals_result["train"]["auc"][best_iter-1]
        eval_score = evals_result["valid"]["auc"][best_iter-1]
        print(f"Train AUC: {train_score:.5f}")
        print(f"Valid AUC: {eval_score:.5f}")

        del model, X_train, y_train, X_valid, y_valid
        gc.collect()

        t_total_end = now()
        total_runtime = print_duration(
            t_total_start, t_total_end, "Total CV Runtime"
        )
        print(f"Free CPU Mem: {round(free_ram_gib(), 2)} GB")
        print(f"Free GPU Mem: {round(free_vram_gib(), 2)} GB")

        fold_summary = {
            "auc": eval_score,
            "runtime": total_runtime
        }

        for lg in loggers:
            lg.on_fold_end(
                fold_idx,
                axis_name="iter",
                evals_result=evals_result,
                summary=fold_summary
            )

        return eval_score
