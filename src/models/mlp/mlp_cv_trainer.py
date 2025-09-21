import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, IterableDataset, get_worker_info
from torch.optim.lr_scheduler import CosineAnnealingLR
import pyarrow.dataset as ds
from dataclasses import dataclass, field
from typing import Iterable, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
import time
from src.utils.print_duration import print_duration


@dataclass(eq=False)
class ParquetStream(IterableDataset):
    # === 引数（元 __init__ のシグネチャ） ===
    paths: list[str] | str | os.PathLike

    features: Optional[list[str]] = None
    target: str = "target"
    cat_cols: Optional[Iterable[str]] = None
    fold_col: Optional[str] = None
    include_folds: Optional[Iterable[int]] = None
    exclude_folds: Optional[Iterable[int]] = None
    weight_col: Optional[str] = None

    batch_size: int
    batch_rows: int = 200_000
    buffer_size: int = 100_000
    rows_per_epoch: int = None
    extra_exclude_cols: Optional[Iterable[str]] = None
    predict_mode: bool = False
    seed: int = 42

    _epoch: int = field(init=False, default=0, repr=False)

    def __post_init__(self):
        super().__init__()
        self.paths = [
            str(p)
            for p in (
                self.paths
                if isinstance(self.paths, (list, tuple))
                else [self.paths]
            )
        ]
        self.buffer_size = int(self.buffer_size)
        self.cat_cols = list(self.cat_cols or [])
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

        # --- extra exclude colsを除外 ---
        schema = ds.dataset(self.paths, format="parquet").schema
        all_cols = [f.name for f in schema]

        if self.extra_exclude_cols:
            excl = (
                {self.extra_exclude_cols}
                if isinstance(self.extra_exclude_cols, str)
                else set(self.extra_exclude_cols)
            )
            self.features = [c for c in self.features if c not in excl]

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

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def _sharded_paths(self):
        info = get_worker_info()
        if info is None:
            return self.paths
        return self.paths[info.id::info.num_workers]

    def __iter__(self):
        rng = np.random.default_rng(
            self.seed
            + self._epoch
            + (get_worker_info().id if get_worker_info() else 0))
        bufX, bufy, bufw = [], [], []
        emitted = 0

        dataset = ds.dataset(self.paths, format="parquet")
        fexpr = None
        if (not self.predict_mode) and self.fold_col:
            col = ds.field(self.fold_col)
            if self.include_folds is not None:
                fexpr = col.isin(sorted(self.include_folds))
            if self.exclude_folds is not None:
                ex = ~col.isin(sorted(self.exclude_folds))
                fexpr = ex if fexpr is None else (fexpr & ex)

        for path in self._sharded_paths():
            reader = dataset.scanner(
                columns=self._columns,
                batch_size=self.batch_rows,
                filster=fexpr
            ).to_reader()
            for batch in reader:
                tbl = batch.to_pandas()  # 速さ重視なら .to_numpy() 直取りでもOK
                X = tbl[self.features].to_numpy(dtype=np.float32, copy=False)
                y = tbl[self.target].to_numpy(dtype=np.float32, copy=False)
                w = tbl[self.weight_col].to_numpy(dtype=np.float32, copy=False) if self.weight_col else None

                # バッファに積む
                bufX.append(X)
                bufy.append(y)
                if self.weight_col:
                    bufw.append(w)
                # バッファが大きくなり過ぎたら1つにまとめてシャッフル
                if sum(len(a) for a in bufy) >= self.buffer_size:
                    Xb = np.concatenate(bufX)
                    yb = np.concatenate(bufy)
                    wb = np.concatenate(bufw) if self.weight_col else None

                for i0 in range(0, len(yb), self.batch_size):
                    i1 = min(i0 + self.batch_size, len(yb))
                    if self.rows_per_epoch and emitted >= self.rows_per_epoch:
                        return
                    xb = torch.from_numpy(Xb[i0:i1])                # (B, F)
                    ybt = torch.from_numpy(yb[i0:i1]).float()       # (B,)
                    if wb is None:
                        yield xb, ybt
                    else:
                        yield xb, ybt, torch.from_numpy(wb[i0:i1]).float()
                    emitted += (i1 - i0)
                    bufX.clear()
                    bufy.clear()
                    bufw.clear()
        # 余りを吐く
        if bufy:
            Xb = np.concatenate(bufX)
            yb = np.concatenate(bufy)
            wb = np.concatenate(bufw) if self.weight_col else None

            for i0 in range(0, len(yb), self.batch_size):
                i1 = min(i0 + self.batch_size, len(yb))
                if self.rows_per_epoch and emitted >= self.rows_per_epoch:
                    return
                xb = torch.from_numpy(Xb[i0:i1])                # (B, F)
                ybt = torch.from_numpy(yb[i0:i1]).float()       # (B,)
                if wb is None:
                    yield xb, ybt
                else:
                    yield xb, ybt, torch.from_numpy(wb[i0:i1]).float()
                emitted += (i1 - i0)


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


class MLPCVTrainer:
    """
    MLPを使ったCVトレーナー。

    Attributes
    ----------
    tr_df : pd.DataFrame
        学習用データ
    test_df : pd.DataFrame, default None
        ラベルなしデータ
    params : dict, default None
        Parameters
    n_splits : int, default 5
        KFoldの分割数
    seed : int, default 42
        乱数シード
    """

    def __init__(
        self,
        tr_df,
        test_df=None,
        params=None,
        n_splits=5,
        seed=42
    ):
        self.params = params or {}
        self.n_splits = n_splits
        self.seed = seed
        self.fold_models = []
        self.fold_scores = []
        self.oof_score = None

        self.default_params = {
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

        self.params = {**self.default_params, **self.params}

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

        if "weight" in tr_df.columns:
            self.weights = tr_df["weight"].to_numpy(dtype=np.float32)
            tr_df = tr_df.drop("weight", axis=1)
        else:
            self.weights = np.ones(len(tr_df), dtype=np.float32)

        self.cat_cols = tr_df.select_dtypes(
            include=["object", "category"]).columns.tolist()
        self.num_cols = [col for col in tr_df.columns
                         if col not in self.cat_cols + ["target"]]

        self.cat_idxs = [tr_df.columns.get_loc(col) for col in self.cat_cols]
        self.num_idxs = [tr_df.columns.get_loc(col) for col in self.num_cols]
        self.cat_dims = [tr_df[col].nunique() for col in self.cat_cols]

        self.X = tr_df.drop("target", axis=1).to_numpy(dtype=np.float32)
        self.y = tr_df["target"].to_numpy(dtype=np.float32)

        if test_df is not None:
            self.test = test_df.to_numpy(dtype=np.float32)
        else:
            self.test = None

        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.seed
        )
        self.fold_indices = list(skf.split(self.X, self.y))

        torch.cuda.manual_seed(self.seed)

    def fit(self):
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
            raise ValueError("test_df not provided for MLPCVTrainer.")

        oof_preds = np.zeros(len(self.X))
        test_preds = np.zeros(len(self.test))

        epoch_list = []

        for fold, (tr_idx, val_idx) in enumerate(self.fold_indices):
            print(f"\nFold {fold + 1}")
            start = time.time()

            X_tr, y_tr, w_tr = (
                self.X[tr_idx],
                self.y[tr_idx],
                self.weights[tr_idx])
            X_val, y_val = (
                self.X[val_idx],
                self.y[val_idx])

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
            test_dataset = TensorDataset(
                torch.tensor(self.test).float()
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.params["batch_size"],
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.params["batch_size"],
                shuffle=False
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.params["batch_size"],
                shuffle=False
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

            best_log_loss = float("inf")
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
                    (epoch - best_epoch >= self.params["early_stopping_rounds"]) and
                    (epoch + 1 >= self.params["min_epochs"])
                ):
                    print(f"Early stopping at epoch {epoch+1}")
                    print(f"Loading best model from epoch {best_epoch} "
                          f"with Logloss {best_log_loss:.5f}")
                    break

            model.load_state_dict(
                {k: v.to(self.params["device"]) for k, v in best_model_state.items()}
            )

            self.fold_scores.append(best_log_loss)

            epoch_list.append(best_epoch)

            model.eval()
            val_preds = []
            with torch.no_grad():
                for xb, _ in val_loader:
                    xb = xb.to(self.params["device"])
                    val_logits = model(xb)
                    val_probs = torch.sigmoid(val_logits).cpu().numpy()
                    val_preds.append(val_probs)
            oof_preds[val_idx] = np.concatenate(val_preds).ravel()

            with torch.no_grad():
                fold_test_preds = []
                for xb in test_loader:
                    xb = xb[0].to(self.params["device"])
                    test_logits = model(xb)
                    test_probs = torch.sigmoid(test_logits).cpu().numpy()
                    fold_test_preds.append(test_probs)
                test_preds += np.concatenate(fold_test_preds).ravel()

            end = time.time()
            print(f"Best Logloss: {best_log_loss:.5f}")
            print_duration(start, end)

        self.oof_score = roc_auc_score(self.y, oof_preds)
        print("\n=== CV Results ===")
        print(f"Fold scores: {self.fold_scores}")
        print(
            f"Mean: {np.mean(self.fold_scores):.5f}, "
            f"Std: {np.std(self.fold_scores):.5f}"
        )
        print(f"OOF score: {self.oof_score:.5f}")
        print(f"Avg best epoch: {np.mean(epoch_list)}")
        print(f"Best epochs: \n{epoch_list}")

        test_preds /= self.n_splits
        return oof_preds, test_preds

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
