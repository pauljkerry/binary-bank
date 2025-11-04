import gc
import os
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional
from time import perf_counter as now

import cudf
import rmm
import torch
import cupy as cp
import rmm.mr as mr
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch.nn as nn
import torch.nn.functional as F

from rmm.allocators.cupy import rmm_cupy_allocator
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.utils.dlpack import from_dlpack as torch_from_dlpack
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.loggers import CVLogger, NoOpLogger
from src.utils.print_duration import print_duration
from src.utils.mem_info import free_ram_gib, free_vram_gib


def compute_feature_stats(
    paths: list[str],
    features: list[str],
    num_cols: list[str],
    fold_col: str = None,
):
    lf = pl.scan_parquet(paths, low_memory=True)

    exprs = []
    for c in num_cols:
        exprs += [pl.col(c).cast(pl.Float32).mean().alias(f"{c}_mean"),
                  pl.col(c).cast(pl.Float32).std(ddof=0).alias(f"{c}_std")]
    out = lf.select(exprs).collect(engine="streaming")
    mean = out.select(
        [f"{c}_mean" for c in num_cols]).to_numpy().ravel().astype(np.float32)
    std = out.select(
        [f"{c}_std" for c in num_cols]).to_numpy().ravel().astype(np.float32)
    std[std < 1e-4] = 1.0
    return mean, std


@dataclass
class ParquetStream(IterableDataset):
    paths: list[str] | str | os.PathLike

    features: list[str]
    target: str
    num_idxs: Iterable[int]

    mean: np.ndarray
    std: np.ndarray

    fold_col: Optional[str] = None
    include_fold: int = None
    exclude_fold: int = None
    weight_col: Optional[str] = None

    batch_size: int = 1024
    rows_per_epoch: int | None = None
    predict_mode: bool = False
    seed: int = 42
    shuffle: bool = True

    _epoch: int = field(init=False, default=0, repr=False)

    def __post_init__(self):
        super().__init__()

        # 形式正規化
        self.paths = [
            str(p)
            for p in (
                self.paths
                if isinstance(self.paths, (list, tuple))
                else [self.paths]
            )
        ]

        # --- スキーマ取得は ParquetFile から（dataset 不使用）---
        pf0 = pq.ParquetFile(self.paths[0])
        all_cols = pf0.schema_arrow.names

        # 入力列（重複除去）
        cols = list(self.features or [])
        if (
            (not self.predict_mode)
            and (self.target in all_cols)
           ):
            cols.append(self.target)
        if (
            (not self.predict_mode)
            and self.weight_col
            and (self.weight_col in all_cols)
        ):
            cols.append(self.weight_col)
        if (
            (not self.predict_mode)
            and self.fold_col
            and (self.fold_col in all_cols)
        ):
            cols.append(self.fold_col)

        self._columns = list(dict.fromkeys(cols))

        self._norm_idxs = cp.asarray(
            self.num_idxs, dtype=cp.int64
        )
        if not (len(self.mean) == len(self.std) == len(self._norm_idxs)):
            raise ValueError(
                f"mean/std/num_idxs length mismatch: "
                f"{len(self.mean)}, {len(self.std)}, {len(self._norm_idxs)}"
            )
        self._mean_cu = cp.asarray(
            self.mean,
            dtype=cp.float32
        )
        self._std_cu = cp.asarray(
            self.std,
            dtype=cp.float32
        )

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def _sharded_paths(self):
        info = get_worker_info()
        if info is None:
            return self.paths
        return self.paths[info.id::info.num_workers]

    def __iter__(self):
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        emitted = 0

        for path in self._sharded_paths():
            pf = pq.ParquetFile(path)
            seed = self.seed + self._epoch + worker_id
            if self.shuffle:
                rg_order = cp.asnumpy(
                    cp.random.RandomState(seed).permutation(pf.num_row_groups)
                )
            else:
                rg_order = np.arange(pf.num_row_groups, dtype=np.int64)

            carry_X = carry_y = carry_w = None

            for rg in rg_order:
                gdf = cudf.read_parquet(
                    path,
                    columns=self._columns,
                    row_groups=[int(rg)]
                ).astype("float32")
                if len(gdf) == 0:
                    continue

                if self.include_fold is not None:
                    gdf = gdf[gdf[self.fold_col] == self.include_fold]
                if self.exclude_fold is not None:
                    gdf = gdf[gdf[self.fold_col] != self.exclude_fold]

                # GPU 内シャッフル
                if self.shuffle:
                    perm = cp.random.RandomState(seed).permutation(len(gdf))
                    gdf = gdf.take(cudf.Series(perm))

                # CuPy へ（ゼロコピー）
                X_cu = gdf[self.features].astype("float32").to_cupy()
                y_cu = (
                    gdf[self.target].astype("float32").values
                    if not self.predict_mode else None
                )
                w_cu = (
                    gdf[self.weight_col].astype("float32").values
                    if (self.weight_col and self.weight_col in gdf.columns)
                    else None
                )

                # 標準化（GPU, in-place）
                if self._norm_idxs.size > 0:
                    ni = self._norm_idxs
                    X_cu[:, ni] -= self._mean_cu
                    X_cu[:, ni] /= (self._std_cu + 1e-8)

                # 端数 carry を前段に連結（必要最小限）
                if carry_X is not None:
                    X_cu = cp.concatenate([carry_X, X_cu], axis=0)
                    if y_cu is not None:
                        y_cu = cp.concatenate([carry_y, y_cu], axis=0)
                    if w_cu is not None:
                        w_cu = cp.concatenate([carry_w, w_cu], axis=0)
                    carry_X = carry_y = carry_w = None

                m = X_cu.shape[0]
                full = (m // self.batch_size) * self.batch_size

                # バッチ生成
                for i in range(0, full, self.batch_size):
                    xb = X_cu[i:i+self.batch_size]
                    if self.predict_mode:
                        yield torch_from_dlpack(xb)
                    else:
                        yb = y_cu[i:i+self.batch_size]
                        if w_cu is not None:
                            wb = w_cu[i:i+self.batch_size]
                            yield (torch_from_dlpack(xb),
                                   torch_from_dlpack(yb).float(),
                                   torch_from_dlpack(wb).float())
                        else:
                            yield (torch_from_dlpack(xb),
                                   torch_from_dlpack(yb).float())
                    emitted += xb.shape[0]
                    if self.rows_per_epoch and emitted >= self.rows_per_epoch:
                        return

                # 端数 carry
                rem = m - full
                if rem:
                    carry_X = X_cu[full:]
                    carry_y = y_cu[full:] if y_cu is not None else None
                    carry_w = w_cu[full:] if w_cu is not None else None

                # 後始末
                del gdf, X_cu
                if y_cu is not None:
                    del y_cu
                if w_cu is not None:
                    del w_cu
                gc.collect()

            # 最後の端数
            if carry_X is not None:
                if self.rows_per_epoch and emitted >= self.rows_per_epoch:
                    return
                if self.predict_mode:
                    yield torch_from_dlpack(carry_X)
                else:
                    if carry_w is not None:
                        yield (torch_from_dlpack(carry_X),
                               torch_from_dlpack(carry_y).float(),
                               torch_from_dlpack(carry_w).float())
                    else:
                        yield (torch_from_dlpack(carry_X),
                               torch_from_dlpack(carry_y).float())


class SimpleMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        dropout_rate,
        activation,
        num_idxs,
        cat_idxs,
        cat_dims
    ):
        super().__init__()
        self.num_idxs = num_idxs
        self.cat_idxs = cat_idxs

        self.embedding_layers = nn.ModuleList([
            nn.Embedding(
                num_embeddings=n, embedding_dim=min(50, (n + 1) // 2))
            for n in cat_dims
        ])

        total_embedding_dim = sum(
            min(50, (n + 1) // 2) for n in cat_dims
        )
        net_input_dim = len(num_idxs) + total_embedding_dim

        layers = []
        prev_dim = net_input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, xb):
        emb_list = [
            self.embedding_layers[i](xb[:, cat_idx].long())
            for i, cat_idx in enumerate(self.cat_idxs)
        ]
        x_emb = torch.cat(emb_list, dim=1) if emb_list else None

        # 数値部分
        x_num = xb[:, self.num_idxs]

        # 結合
        if x_emb is not None:
            x = torch.cat([x_num, x_emb], dim=1)
        else:
            x = x_num

        return self.net(x).squeeze(-1)


@dataclass
class MLPCVTrainer:
    data_id: int
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
            "lr": 1e-3,
            "batch_size": 256,
            "dropout_rate": 0.2,
            "hidden_dim1": 128,
            "hidden_dim2": 64,
            "hidden_dim3": None,
            "hidden_dim4": None,
            "activation": "ReLU",
            "early_stopping_rounds": 10,
            "t_max": 50,
            "eta_min": 1e-6,
            "device": "cuda"
        }

        ACTIVATION_MAPPING = {
            "ReLU": nn.ReLU,
            "LeakyReLU": nn.LeakyReLU,
            "ELU": nn.ELU,
            "GELU": nn.GELU,
            "SiLU": nn.SiLU,
            "Tanh": nn.Tanh,
            "Sigmoid": nn.Sigmoid,
        }

        self.params = {**default_params, **self.params}

        self.params["activation"] = ACTIVATION_MAPPING[self.params["activation"]]

        self.params["max_epochs"] = self.opts.get("max_epochs", 100)
        self.params["min_epochs"] = self.opts.get("min_epochs", 20)

        hidden_dims = []
        i = 1
        while f"hidden_dim{i}" in self.params:
            dim = self.params[f"hidden_dim{i}"]
            if dim is None or dim == -1:
                break
            hidden_dims.append(dim)
            i += 1

        self.params["hidden_dims"] = hidden_dims

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

        self.num_cols = [
            col for col in self.features
            if col not in self.cat_cols
        ]

        self.cat_idxs = [self.features.index(c) for c in self.cat_cols]
        self.num_idxs = [self.features.index(c) for c in self.num_cols]

        scan = pl.scan_parquet(self.train_paths)
        exprs = [
            pl.col(c)
            .rank("dense")
            .cast(pl.Int32)
            .n_unique()
            .alias(c) for c in self.cat_cols
        ]
        df1 = scan.select(exprs).collect()

        if df1.width == 0 or df1.height == 0:
            self.cat_dims = []
        else:
            self.cat_dims = [int(x) if x is not None else 0 for x in df1.row(0)]

        torch.cuda.manual_seed(self.seed)

        self.mean, self.std = compute_feature_stats(
            self.train_paths,
            self.features,
            self.num_cols
        )

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
        one_fold: bool = True
    ) -> dict:
        t_total_start = now()

        loggers = loggers or [NoOpLogger()]
        meta = {
            "data_id": self.data_id,
            "seed": self.seed,
            "n_folds": self.n_folds,
            **self.params
        }
        for lg in loggers:
            lg.on_start(meta)

        if not one_fold:
            train_rows = pq.ParquetFile(self.train_paths[0]).metadata.num_rows
            test_rows = pq.ParquetFile(self.test_paths[0]).metadata.num_rows

            oof = np.zeros(train_rows, dtype=np.float32)
            test_pred = np.zeros(test_rows, dtype=np.float32)

        epoch_list = []
        fold_scores = {name: [] for name in self.metrics.keys()}

        fold_df = (
            pl.read_parquet(
                self.train_paths,
                columns=["row_id", self.fold_col]
            )
        )

        for i in range(self.n_folds):
            title = f" Fold {i + 1} / {self.n_folds} "
            print("=" * 48)
            print(f"{title:=^48}")
            print("=" * 48)
            print(f"Free CPU Mem: {round(free_ram_gib(), 2)} GB")
            print(f"Free GPU Mem: {round(free_vram_gib(), 2)} GB")

            t_fold_start = now()
            fold_summary = {}

            train_ds = ParquetStream(
                self.train_paths,
                self.features,
                self.target,
                self.num_idxs,
                self.mean,
                self.std,
                fold_col=self.fold_col,
                exclude_fold=i,
                weight_col=self.weight_col,
                batch_size=self.params["batch_size"],
                predict_mode=False,
                seed=self.seed,
                shuffle=True
            )
            valid_ds = ParquetStream(
                self.train_paths,
                self.features,
                self.target,
                self.num_idxs,
                self.mean,
                self.std,
                fold_col=self.fold_col,
                include_fold=i,
                batch_size=self.params["batch_size"],
                predict_mode=False,
                seed=self.seed,
                shuffle=False
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=None,
                num_workers=0,
                shuffle=False
            )
            val_loader = DataLoader(
                valid_ds,
                batch_size=None,
                num_workers=0,
                shuffle=False
            )

            if not one_fold:
                test_ds = ParquetStream(
                    self.test_paths,
                    self.features,
                    self.target,
                    self.num_idxs,
                    self.mean,
                    self.std,
                    fold_col=self.fold_col,
                    batch_size=self.params["batch_size"],
                    predict_mode=True,
                    seed=self.seed,
                    shuffle=False
                )
                test_loader = DataLoader(
                    test_ds,
                    batch_size=None,
                    num_workers=0,
                    shuffle=False
                )

            lf = pl.scan_parquet(self.train_paths)
            y_val = (
                lf.filter(pl.col(self.fold_col) == i)
                .select(self.target)
                .collect(engine="streaming")
                .to_series()
                .to_numpy()
                .astype("float32")
            )
            val_idx = (
                fold_df
                .filter(pl.col(self.fold_col) == i)
                .get_column("row_id")
                .to_numpy()
                .astype(np.int32, copy=False)
            )

            model = SimpleMLP(
                input_dim=len(self.features),
                hidden_dims=self.params["hidden_dims"],
                dropout_rate=self.params["dropout_rate"],
                activation=self.params["activation"],
                num_idxs=self.num_idxs,
                cat_idxs=self.cat_idxs,
                cat_dims=self.cat_dims
            ).to(self.params["device"])

            optimizer = torch.optim.Adam(model.parameters(), lr=self.params["lr"])
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.params["t_max"],
                eta_min=self.params["eta_min"]
            )

            best_logloss = float("inf")
            best_model_state = None
            best_epoch = 0

            history = {
                "train": {key: [] for key in self.metrics},
                "valid": {key: [] for key in self.metrics}
            }
            extra_hist = {"lr": []}

            for epoch in range(self.params["max_epochs"]):
                model.train()
                for batch in train_loader:
                    if len(batch) == 3:
                        xb, yb, wb = batch
                    else:
                        xb, yb = batch
                        wb = None

                    preds = model(xb)

                    if wb is None:
                        loss = F.binary_cross_entropy_with_logits(
                            preds, yb, reduction="mean"
                        )
                    else:
                        loss = F.binary_cross_entropy_with_logits(
                            preds, yb, weight=wb, reduction="mean"
                        )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Validation
                model.eval()
                val_pred = []
                train_pred = []
                y_train = []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        pred_logits = model(xb)
                        pred_probs = torch.sigmoid(pred_logits).cpu().numpy()
                        val_pred.append(pred_probs)

                    for batch in train_loader:
                        if len(batch) == 3:
                            xb, yb, wb = batch
                        else:
                            xb, yb = batch
                            wb = None
                        xb = xb.to(self.params["device"])
                        pred_logits = model(xb)
                        pred_probs = torch.sigmoid(
                            pred_logits).cpu().numpy()
                        train_pred.append(pred_probs)
                        y_train.append(yb.cpu().numpy())

                val_pred = np.concatenate(val_pred)
                train_pred = np.concatenate(train_pred)
                y_train = np.concatenate(y_train)

                for name, metric_func in self.metrics.items():
                    train_score = metric_func(y_train, train_pred)
                    val_score = metric_func(y_val, val_pred)
                    history["train"][name].append(train_score)
                    history["valid"][name].append(val_score)

                lr = scheduler.get_last_lr()[0]
                extra_hist["lr"].append(lr)
                scheduler.step()

                logloss_train = history['train']['log_loss'][-1]
                logloss_val = history['valid']['log_loss'][-1]
                print(
                    f"Epoch {epoch+1}: "
                    f"Train LogLoss = {logloss_train:.5f}, "
                    f"Val LogLoss = {logloss_val:.5f}"
                )

                if logloss_val < best_logloss:
                    best_logloss = logloss_val
                    best_model_state = {
                        k: v.cpu().clone() for k, v
                        in model.state_dict().items()
                    }
                    best_epoch = epoch + 1
                    print(
                        f"New best model saved at epoch {epoch+1}, "
                        f"Logloss: {logloss_val:.5f}")
                elif (
                    (epoch - best_epoch >= self.params["early_stopping_rounds"])
                    and (epoch + 1 >= self.params["min_epochs"])
                ):
                    print(f"Early stopping at epoch {epoch+1}")
                    print(f"Loading best model from epoch {best_epoch} "
                          f"with Logloss {logloss_val:.5f}")
                    break

            model.load_state_dict(
                {
                    k: v.to(self.params["device"])
                    for k, v in best_model_state.items()
                }
            )

            model.eval()
            val_pred = []
            fold_test_pred = []
            with torch.no_grad():
                for xb, _ in val_loader:
                    xb = xb.to(self.params["device"])
                    val_logits = model(xb)
                    val_probs = torch.sigmoid(val_logits).cpu().numpy()
                    val_pred.append(val_probs)

                if not one_fold:
                    for xb in test_loader:
                        xb = xb.to(self.params["device"])
                        test_logits = model(xb)
                        test_probs = torch.sigmoid(test_logits).cpu().numpy()
                        fold_test_pred.append(test_probs)

            val_pred = np.concatenate(val_pred).ravel()

            for name, metric_func in self.metrics.items():
                val_score = metric_func(y_val, val_pred)
                print(f"{name.upper()} Valid: {val_score:.5f}")
                fold_summary[name] = val_score
                fold_scores[name].append(val_score)

            t_fold_end = now()
            fold_summary["runtime"] = print_duration(
                t_fold_start, t_fold_end, f"\nFold {i+1} Runtime"
            )

            for lg in loggers:
                lg.on_fold_end(
                    i,
                    "epoch",
                    history,
                    extra_hist,
                    fold_summary
                )

            if one_fold:
                result = {
                    "oof": None,
                    "test_pred": None,
                    "oof_score": fold_scores[self.rep_metric][0]
                }
            else:
                oof[val_idx] = val_pred
                test_pred += np.concatenate(fold_test_pred).ravel()

            del model
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

        epoch_mean = np.mean(epoch_list)

        print(f"\n{' CV Results ':*^48}")
        print("─" * 48)
        print(f" {'Metric':^9}  {'OOF':>10}  {'Mean':>10} ± {'Std':<10} ")
        print("-" * 48)
        for name, stats in oof_stats.items():
            print(f" {name.upper():^9} "
                  f" {stats['oof']:>10.5f} "
                  f" {stats['mean']:>10.5f} ± {stats['std']:<10.5f} ")
        print("─" * 48)

        print(f"Avg best epoch: {epoch_mean}")

        print(f"Free CPU Mem: {round(free_ram_gib(), 2)} GB")
        print(f"Free GPU Mem: {round(free_vram_gib(), 2)} GB")

        result = {
            "oof": oof,
            "test_pred": test_pred,
            "oof_score": oofs[self.rep_metric]
        }
        overall_summary = {"epoch_mean": epoch_mean}
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
