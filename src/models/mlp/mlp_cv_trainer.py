import gc
import os
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional
from pathlib import Path
from time import perf_counter as now

import cudf
import torch
import cupy as cp
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, IterableDataset, get_worker_info
from torch.utils.dlpack import from_dlpack
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.loggers import CVResult, CVLogger, NoOpLogger
from src.utils.print_duration import print_duration
from src.utils.win_avail_gb import win_avail_gb


def compute_feature_stats(
    paths: list[str],
    features: list[str],
    num_cols: list[str],
    fold_col: str = None,
    include_folds: str = None,
    exclude_folds: str = None,
    batch_rows: int = 1_000_000,
):
    lf = pl.scan_parquet(paths, low_memory=True)
    if fold_col:
        if include_folds is not None:
            lf = lf.filter(pl.col(fold_col).is_in(sorted(include_folds)))
        if exclude_folds is not None:
            lf = lf.filter(~pl.col(fold_col).is_in(sorted(exclude_folds)))

    exprs = []
    for c in num_cols:
        exprs += [pl.col(c).cast(pl.Float64).mean().alias(f"{c}_mean"),
                  pl.col(c).cast(pl.Float64).std(ddof=0).alias(f"{c}_std")]
    out = lf.select(exprs).collect(streaming=True)  # 大規模でも低メモリ
    mean = out.select(
        [f"{c}_mean" for c in num_cols]).to_numpy().ravel().astype(np.float32)
    std = out.select(
        [f"{c}_std" for c in num_cols]).to_numpy().ravel().astype(np.float32)
    std[std == 0] = 1.0
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
    include_folds: Optional[Iterable[int]] = None
    exclude_folds: Optional[Iterable[int]] = None
    weight_col: Optional[str] = None

    batch_size: int = 1024
    rows_per_epoch: int | None = None
    extra_exclude_cols: Optional[Iterable[str]] = None
    predict_mode: bool = False
    seed: int = 42

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
        self.num_idxs = list(self.num_idxs or [])
        self.include_folds = (
            None
            if self.include_folds is None
            else set(self.include_folds)
        )
        self.exclude_folds = (
            None
            if self.exclude_folds is None
            else set(self.exclude_folds)
        )
        self.predict_mode = bool(self.predict_mode)

        # --- スキーマ取得は ParquetFile から（dataset 不使用）---
        pf0 = pq.ParquetFile(self.paths[0])
        all_cols = pf0.schema_arrow.names

        if self.extra_exclude_cols:
            excl = (
                {self.extra_exclude_cols}
                if isinstance(self.extra_exclude_cols, str)
                else set(self.extra_exclude_cols)
            )
            self.features = [c for c in self.features if c not in excl]

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
            rg_order = cp.asnumpy(
                cp.random.RandomState(seed).permutation(pf.num_row_groups)
            )

            carry_X = carry_y = carry_w = None

            for rg in rg_order:
                gdf = cudf.read_parquet(
                    path,
                    columns=self._columns,
                    row_groups=[int(rg)]
                )
                if len(gdf) == 0:
                    continue

                # fold フィルタ（GPU）
                if (not self.predict_mode) and self.fold_col and (self.fold_col in gdf.columns):
                    if self.include_folds is not None:
                        gdf = gdf[gdf[self.fold_col].isin(sorted(self.include_folds))]
                    if self.exclude_folds is not None:
                        gdf = gdf[~gdf[self.fold_col].isin(sorted(self.exclude_folds))]
                    if len(gdf) == 0:
                        continue

                # GPU 内シャッフル
                perm = cp.random.RandomState(seed).permutation(len(gdf))
                gdf = gdf.take(cudf.Series(perm))

                # CuPy へ（ゼロコピー）
                X_cu = gdf[self.features].to_cupy()
                y_cu = gdf[self.target].values if not self.predict_mode else None
                w_cu = (
                    gdf[self.weight_col].values
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
                        yield from_dlpack(xb.toDlpack())
                    else:
                        yb = y_cu[i:i+self.batch_size]
                        if w_cu is not None:
                            wb = w_cu[i:i+self.batch_size]
                            yield (from_dlpack(xb.toDlpack()),
                                   from_dlpack(yb.toDlpack()).float(),
                                   from_dlpack(wb.toDlpack()).float())
                        else:
                            yield (from_dlpack(xb.toDlpack()),
                                   from_dlpack(yb.toDlpack()).float())
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
                    yield from_dlpack(carry_X.toDlpack())
                else:
                    if carry_w is not None:
                        yield (from_dlpack(carry_X.toDlpack()),
                               from_dlpack(carry_y.toDlpack()).float(),
                               from_dlpack(carry_w.toDlpack()).float())
                    else:
                        yield (from_dlpack(carry_X.toDlpack()),
                               from_dlpack(carry_y.toDlpack()).float())


class SimpleMLP(nn.Module):
    """
    Parameters
    ----------
    input_dim : int
        数値特徴量の次元数（カテゴリ変数を除いたもの）
    hidden_dims : list of int
        MLPの隠れ層サイズリスト
    dropout_rate : float
        ドロップアウト率
    activation : nn.Module
        活性化関数
    num_idxs : list
        数値変数のインデックスのリスト
    cat_idxs : list
        カテゴリ変数のインデックスのリスト
    cat_dims : list
        カテゴリ変数のユニークな値のリスト
    """

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
        """
        Parameters
        ----------
        xb : torch.Tensor
            数値特徴量とカテゴリ特徴量を統合したもの

        Returns
        -------
        torch.Tensor
            (B,) の予測
        """
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
    """
    MLPを使ったCVトレーナー。

    Attributes
    ----------
    data_id : int
        ID for dataset version
    train_paths : str or list[str]
        Path(s) to training parquet files.
    test_paths : str or list[str]
        Path(s) to test parquet files.
    features : Optional[list[str]], default None
        name of column for training
    target : str, default "target"
        name of column for target.
    seed : int, default 42
        乱数シード
    """
    data_id: int
    train_paths: str | list[str]
    test_paths: str | list[str] | None = None

    features: Optional[list[str]] = None

    target: str = "target"
    fold_col: Optional[str] = None
    weight_col: Optional[str] = None
    cat_cols: Optional[list[str]] = None

    params: dict = field(default_factory=dict)

    n_fold: int = 5
    seed: int = 42
    gpu: bool = True

    opts: dict = field(init=True, default_factory=dict)

    def __post_init__(self):
        self.feature_dir = Path(self.feature_dir)

        default_params = {
            "lr": 1e-3,
            "batch_size": 256,
            "dropout_rate": 0.2,
            "hidden_dim1": 128,
            "hidden_dim2": 64,
            "hidden_dim3": None,
            "hidden_dim4": None,
            "max_epochs": 100,
            "min_epochs": 30,
            "activation": "ReLU",
            "early_stopping_rounds": 10,
            "t_max": 50,
            "eta_min": 1e-6,
            "log_interval": 1,
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

        hidden_dims = []
        i = 1
        while f"hidden_dim{i}" in self.params:
            dim = self.params[f"hidden_dim{i}"]
            if dim is None or dim == -1:
                break
            hidden_dims.append(dim)
            i += 1

        self.params["hidden_dims"] = hidden_dims

        hdr = pl.read_parquet(self.train_paths, n_rows=0)
        all_cols = hdr.columns

        if self.fold_col is None:
            self.fold_col = f"{self.n_fold}fold-s{self.seed}"

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

        self.num_cols = [
            col for col in self.features
            if col not in self.cat_cols
        ]

        self.cat_idxs = [self.features.index(c) for c in self.cat_cols]
        self.num_idxs = [self.features.index(c) for c in self.num_cols]

        scan = pl.scan_parquet(self.train_paths, columns=self.cat_cols)
        exprs = [pl.col(c).n_unique().alias(c) for c in self.cat_cols]
        df1 = scan.select(exprs).collect()
        self.cat_dims = list(df1.row(0))

        torch.cuda.manual_seed(self.seed)

    def fit(
        self,
        extra_exclude_cols=None,
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
        if self.test is None:
            raise ValueError("Please provide test_paths (got None).")

        t_total_start = now()

        loggers = loggers or [NoOpLogger()]
        meta = {
            "data_id": self.data_id,
            "seed": self.seed,
            "n_fold": self.n_fold,
            "batch_rows": self.batch_rows,
            **self.params
        }
        for lg in loggers:
            lg.on_start(meta)

        train_rows = pq.ParquetFile(self.train_path).metadata.num_rows
        test_rows = pq.ParquetFile(self.test_path).metadata.num_rows

        oof = np.zeros(train_rows, dtype=np.float32)
        test_pred = np.zeros(test_rows, dtype=np.float32)

        epoch_list = []
        fold_scores = []

        for i in range(self.n_fold):
            print("=" * 22)
            print(f"===== Fold {i + 1} / {self.n_fold} =====")
            print("=" * 22)
            print(f"Avail Mem: {round(win_avail_gb(), 2)} GB")

            t_fold_start = now()

            mean, std = compute_feature_stats(
                self.train_patahs,
                self.features,
                self.num_cols,
                self.fold_col,
                exclude_folds=i
            )

            train_ds = ParquetStream(
                self.train_paths,
                self.features,
                self.target,
                mean,
                std,
                self.fold_col,
                exclude_folds=i,
                weight_col=self.weight_col,
                batch_seize=self.batch_size,
                buffer_size=self.buffer_size,
                extra_exclude=extra_exclude_cols,
                predict_mode=False,
                seed=self.seed
            )
            valid_ds = ParquetStream(
                self.train_paths,
                self.features,
                self.target,
                mean,
                std,
                self.cat_cols,
                self.fold_col,
                include_folds=i,
                batch_seize=self.batch_size,
                buffer_size=self.buffer_size,
                extra_exclude=extra_exclude_cols,
                predict_mode=False,
                seed=self.seed
            )
            test_ds = ParquetStream(
                self.train_paths,
                self.features,
                self.target,
                mean,
                std,
                self.cat_cols,
                batch_seize=self.batch_size,
                buffer_size=self.buffer_size,
                extra_exclude=extra_exclude_cols,
                predict_mode=True,
                seed=self.seed,
            )

            train_loader = DataLoader(
                train_ds,
                batch_size=None,
                shuffle=False
            )
            val_loader = DataLoader(
                valid_ds,
                batch_size=None,
                shuffle=False
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=None,
                shuffle=False
            )

            lf = pl.scan_parquet(
                self.train_paths, columns=["row_id", self.target]
            )
            y_val = (
                lf.filter(pl.col(self.fold_col) == i)
                .select(self.target)
                .collect(streaming=True)
                .to_series()
                .to_numpy()
                .astype("float32")
                    )

            val_idx = (
                lf.filter(pl.col(self.fold_col) == i)
                  .select("row_id")
                  .collect(engine="streaming")
                  .to_series()
                  .to_numpy()
                  .astype(np.int32, copy=False)
            )

            evals_result = {}

            model = SimpleMLP(
                input_dim=self.X.shape[1],
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

            best_log_loss = float("inf")
            best_model_state = None
            best_epoch = 0

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
                preds = []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        pred_logits = model(xb)
                        pred_probs = torch.sigmoid(pred_logits).cpu().numpy()
                        preds.append(pred_probs)
                val_pred = np.concatenate(preds)
                val_log_loss = log_loss(y_val, val_pred)
                scheduler.step()

                if (epoch + 1) % self.params["log_interval"] == 0 or epoch == 0:
                    model.eval()
                    train_preds = []
                    train_targets = []
                    with torch.no_grad():
                        for xb, yb, wb in train_loader:
                            xb = xb.to(self.params["device"])
                            pred_logits = model(xb)
                            pred_probs = torch.sigmoid(
                                pred_logits).cpu().numpy()
                            train_preds.append(pred_probs)
                            train_targets.append(yb.numpy())
                    train_preds = np.concatenate(train_preds)
                    train_targets = np.concatenate(train_targets)
                    train_log_loss = log_loss(train_targets, train_preds)

                    print(
                        f"Epoch {epoch+1}: "
                        f"Train Logloss = {train_log_loss:.5f}, "
                        f"Val Logloss = {val_log_loss:.5f}"
                    )

                train_logloss_list.append(train_log_loss)
                evals_logloss_list.append(val_log_loss)

                if val_log_loss < best_log_loss:
                    best_log_loss = val_log_loss
                    best_model_state = {
                        k: v.cpu().clone() for k, v
                        in model.state_dict().items()
                    }
                    best_epoch = epoch + 1
                    print(
                        f"New best model saved at epoch {epoch+1}, "
                        f"Logloss: {val_log_loss:.5f}")
                elif (
                    (epoch - best_epoch >= self.params["early_stopping_rounds"])
                    and (epoch + 1 >= self.params["min_epochs"])
                ):
                    print(f"Early stopping at epoch {epoch+1}")
                    print(f"Loading best model from epoch {best_epoch} "
                          f"with Logloss {best_log_loss:.5f}")
                    break

            model.load_state_dict(
                {k: v.to(self.params["device"]) for k, v in best_model_state.items()}
            )

            fold_scores.append(best_log_loss)

            epoch_list.append(best_epoch)

            model.eval()
            val_preds = []
            with torch.no_grad():
                for xb, _ in val_loader:
                    xb = xb.to(self.params["device"])
                    val_logits = model(xb)
                    val_probs = torch.sigmoid(val_logits).cpu().numpy()
                    val_preds.append(val_probs)
            oof[val_idx] = np.concatenate(val_preds).ravel()

            with torch.no_grad():
                fold_test_preds = []
                for xb in test_loader:
                    xb = xb[0].to(self.params["device"])
                    test_logits = model(xb)
                    test_probs = torch.sigmoid(test_logits).cpu().numpy()
                    fold_test_preds.append(test_probs)
                test_pred += np.concatenate(fold_test_preds).ravel()

            end = time.time()
            print(f"Best Logloss: {best_log_loss:.5f}")
            print_duration(start, end)

            evals_result["train"] = {"logloss": logloss_list, "eta": eta_list}
            evals_result["eval"] = {"logloss": logloss_list, "eta": eta_list}

        self.oof_score = roc_auc_score(self.y, oof)
        print("\n=== CV Results ===")
        print(f"Fold scores: {fold_scores}")
        print(
            f"Mean: {np.mean(fold_scores):.5f}, "
            f"Std: {np.std(fold_scores):.5f}"
        )
        print(f"OOF score: {self.oof_score:.5f}")
        print(f"Avg best epoch: {np.mean(epoch_list)}")
        print(f"Best epochs: \n{epoch_list}")

        test_pred /= self.n_splits
        return oof, test_pred

    def fit_one_fold(self, fold=0):
        """
        指定した1つのfoldのみを用いてモデルを学習する。
        主にOptunaによるハイパーパラメータ探索時に使用。

        Parameters
        ----------
        fold : int
            学習に使うfold番号。

        Rerurn
        ------
        best_logloss : float
            Score
        """
        tr_idx, val_idx = self.fold_indices[fold]
        start = time.time()

        X_tr, y_tr, w_tr = (
            self.X[tr_idx], self.y[tr_idx], self.weights[tr_idx])
        X_val, y_val = self.X[val_idx], self.y[val_idx]

        # Dataloaders
        train_dataset = TensorDataset(
            torch.tensor(X_tr),
            torch.tensor(y_tr),
            torch.tensor(w_tr)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val),
            torch.tensor(y_val)
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.params["batch_size"], shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.params["batch_size"], shuffle=False
        )

        model = SimpleMLP(
            input_dim=self.X.shape[1],
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
        criterion = nn.BCEWithLogitsLoss()

        best_logloss = float("inf")
        best_model_state = None
        best_epoch = 0

        for epoch in range(self.params["max_epochs"]):
            model.train()
            for xb, yb, wb in train_loader:
                xb = xb.to(self.params["device"])
                yb = yb.to(self.params["device"])
                wb = wb.to(self.params["device"])

                preds = model(xb)
                loss = criterion(preds, yb)
                loss = (loss * wb).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            preds = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.params["device"])

                    pred_logits = model(xb)
                    pred_probs = torch.sigmoid(pred_logits).cpu().numpy()
                    preds.append(pred_probs)
            val_pred = np.concatenate(preds)
            val_logloss = log_loss(y_val, val_pred)
            scheduler.step()

            if (epoch + 1) % 1 == 0 or epoch == 0:
                model.eval()

                train_preds = []
                train_targets = []
                with torch.no_grad():
                    for xb, yb, wb in train_loader:
                        xb = xb.to(self.params["device"])

                        pred_logits = model(xb)
                        pred_probs = torch.sigmoid(pred_logits).cpu().numpy()
                        train_preds.append(pred_probs)
                        train_targets.append(yb.numpy())
                train_preds = np.concatenate(train_preds)
                train_targets = np.concatenate(train_targets)
                train_log_loss = log_loss(train_targets, train_preds)

                print(
                    f"Epoch {epoch+1}: "
                    f"Train Logloss = {train_log_loss:.5f}, "
                    f"Val Logloss = {val_logloss:.5f}"
                )

            if val_logloss < best_logloss:
                best_logloss = val_logloss
                best_model_state = model.state_dict()
                best_model_state = {
                    k: v.cpu().clone() for k, v
                    in model.state_dict().items()
                }
                print(
                    f"New best model saved at epoch {epoch+1}, "
                    f"Logloss: {val_logloss:.5f}")
                best_epoch = epoch + 1
            elif (
                (epoch - best_epoch >= self.params["early_stopping_rounds"]) and
                (epoch + 1 >= self.params["min_epochs"])
            ):
                print(f"Early stopping at epoch {epoch+1}")
                print(
                    f"Loading best model from epoch {best_epoch} "
                    f"with Logloss {best_logloss:.5f}")
                break

        model.load_state_dict(
            {k: v.to(self.params["device"]) for k, v in best_model_state.items()}
        )

        model.eval()
        final_preds = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(self.params["device"])
                pred_logits = model(xb)
                pred_probs = torch.sigmoid(pred_logits).cpu().numpy()
                final_preds.append(pred_probs)

        final_preds = np.concatenate(final_preds)
        best_auc = roc_auc_score(y_val, final_preds)

        end = time.time()
        print_duration(start, end)
        print(f"Best Logloss: {best_logloss:.5f}")
        print(f"Best AUC: {best_auc:.5f}")

        return best_auc
