import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
import time
import joblib
from src.utils.print_duration import print_duration


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

    def __init__(self, input_dim, hidden_dims, dropout_rate, activation,
                 num_idxs, cat_idxs, cat_dims):
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
            self.fold_models.append(MLPFoldModel(
                model,
                X_val,
                y_val,
                fold,
                best_rounds=best_epoch
            ))
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


class MLPFoldModel:
    """
    MLPのfold単位のモデルを保持するクラス。

    Attributes
    ----------
    model : torch.nn.Module
        学習済みのMLPモデル。
    X_val : pd.DataFrame
        検証用の特徴量データ。
    y_val : pd.Series
        検証用のターゲットラベル。
    fold_index : int
        foldの番号。
    best_rounds : int
        最良スコア時のエポック数
    """

    def __init__(
        self, model, X_val, y_val, fold_index, best_rounds
    ):
        self.model = model
        self.X_val_num = X_val
        self.y_val = y_val
        self.fold_index = fold_index
        self.best_rounds = best_rounds

    def save_model(self, path="../artifacts/model/xgb_vn.pkl"):
        """
        学習済みモデルを指定パスに保存する。

        Parameters
        ----------
        path : str
            モデルを保存するパス。
        """
        joblib.dump(self.model, path)

    def load_model(self, path):
        """
        指定されたパスからモデルを読み込む。

        Parameters
        ----------
        path : str
            モデルファイルのパス。

        Returns
        -------
        self : XGBFoldModel
            読み込んだモデルを保持するインスタンス自身を返す。
        """
        self.model = joblib.load(path)
        return self