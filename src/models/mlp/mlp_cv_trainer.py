import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import time
import joblib
from src.utils.print_duration import print_duration


class SimpleMLP(nn.Module):
    """
    シンプルな多層パーセプトロン（MLP）モデル。

    3層の全結合層（ReLU活性化 + Dropout）を通じて回帰タスクに対応。

    Parameters
    ----------
    input_dim : int
        入力特徴量の次元数。
    hidden_dims : list of int
        各隠れ層のユニット数。
    dropout_rate : float
        各層に適用するドロップアウト率。
    activation : nn.Module
        活性化関数のクラス
    """

    def __init__(self, input_dim, hidden_dims, dropout_rate, activation):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        入力データに対する順伝播を行い、予測値を返す。

        Parameters
        ----------
        x : torch.Tensor
            形状が (バッチサイズ, 入力次元) の入力テンソル。

        Returns
        -------
        torch.Tensor
            形状が (バッチサイズ,) の出力テンソル。
        """
        return self.net(x).squeeze(-1)  # (B,) にする


class MLPCVTrainer:
    """
    MLPを使ったCVトレーナー。

    Parameters
    ----------
    n_splits : int, default 5
        StratifiedKFoldの分割数。
    seed : int, default 42
        乱数シード。
    epochs : int, default 100
        エポック数。
    early_stopping_rounds : int, default 20
        早期停止ラウンド数
    batch_size : int, default 256
        バッチサイズ。
    lr : float, default 1e-3
        learning rate。
    use_gpu : bool, default True
        Trueの場合はGPUが使用可能であれば使用する。
    hidden_dims : list of int, default [128, 64]
        各隠れ層のユニット数。
    dropout_rate : float, default 0.2
        各層に適用するドロップアウト率。
    activation : str, default "ReLU"
        活性化関数のクラス
    log_interval : int, default 1
        scoreのログの表示頻度
    t_max : int, default 50
        CosineAnnealingLRにおける最大エポック数
    eta_min : float, default 1e-6
        CosineAnnealingLRにおける最小学習率
    min_epochs : int, default 50
        最低限学習するエポック数

    Other Parameters
    ----------------
    hidden_dim1 : int, optional
    hidden_dim2 : int, optional
    hidden_dim3 : int, optional
        各隠れ層のユニット数。"hidden_dims"を指定しない場合に有効。
    """

    def __init__(
        self, n_splits=5, seed=42, epochs=100, early_stopping_rounds=20,
        batch_size=256, lr=1e-3, use_gpu=True, hidden_dims=None,
        dropout_rate=0.2, activation="ReLU", log_interval=1,
        t_max=50, eta_min=1e-6, min_epochs=50, **kwargs
    ):
        self.n_splits = n_splits
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.early_stopping_rounds = early_stopping_rounds
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.fold_models = []
        self.fold_scores = []
        self.oof_score = None
        self.early_stopping_rounds = early_stopping_rounds
        self.dropout_rate = dropout_rate
        self.log_interval = log_interval
        self.t_max = t_max
        self.eta_min = eta_min
        self.min_epochs = min_epochs

        if hidden_dims is not None:
            self.hidden_dims = hidden_dims
        else:
            self.hidden_dims = [
                v for k, v in sorted(kwargs.items())
                if k.startswith("hidden_dim")
            ]

        ACTIVATION_MAPPING = {
            "ReLU": nn.ReLU,
            "LeakyReLU": nn.LeakyReLU,
            "ELU": nn.ELU,
            "GELU": nn.GELU,
            "SiLU": nn.SiLU,
            "Tanh": nn.Tanh,
            "Sigmoid": nn.Sigmoid,
        }
        self.activation = ACTIVATION_MAPPING[activation]

    def fit(self, tr_df, test_df):
        """
        CVを用いてモデルを学習し、OOF予測とtest_dfの平均予測を返す。

        Parameters
        ----------
        tr_df : cudf.DataFrame
            学習用データ。
        test_df : cudf.DataFrame
            テスト用データ。

        Returns
        -------
        oof_preds : ndarray
            OOF予測配列
        test_preds : ndarray
            test_dfに対する予測配列
        """
        tr_df = tr_df.copy()
        test_df = test_df.copy()

        if "weight" in tr_df.columns:
            weights = tr_df["weight"].astype("float32").values
            tr_df = tr_df.drop("weight", axis=1)
        else:
            weights = np.ones(len(tr_df), dtype="float32")

        X = tr_df.drop("target", axis=1).to_numpy().astype(np.float32)
        y = tr_df["target"].to_numpy().astype(np.float32)
        X_test = test_df.to_numpy().astype(np.float32)

        oof_preds = np.zeros(len(X))
        test_preds = np.zeros(len(X_test))

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        epoch_list = []

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\nFold {fold + 1}")
            start = time.time()

            X_tr, y_tr, w_tr = X[tr_idx], y[tr_idx], weights[tr_idx]
            X_val, y_val = X[val_idx], y[val_idx]

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
                train_dataset, batch_size=self.batch_size, shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )

            model = SimpleMLP(
                input_dim=X.shape[1],
                hidden_dims=self.hidden_dims,
                dropout_rate=self.dropout_rate,
                activation=self.activation
            ).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.t_max,
                eta_min=self.eta_min
            )
            criterion = nn.BCEWithLogitsLoss()

            best_log_loss = float("inf")
            best_model_state = None
            best_epoch = 0

            for epoch in range(self.epochs):
                model.train()
                for xb, yb, wb in train_loader:
                    xb, yb, wb = (
                        xb.to(self.device),
                        yb.to(self.device),
                        wb.to(self.device)
                    )
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
                        xb = xb.to(self.device)
                        pred_logits = model(xb)
                        pred_probs = torch.sigmoid(pred_logits).cpu().numpy()
                        preds.append(pred_probs)
                val_pred = np.concatenate(preds)
                val_log_loss = log_loss(y_val, val_pred)
                scheduler.step()

                if (epoch + 1) % self.log_interval == 0 or epoch == 0:
                    model.eval()
                    train_preds = []
                    train_targets = []
                    with torch.no_grad():
                        for xb, yb, wb in train_loader:
                            xb = xb.to(self.device)
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
                    (epoch - best_epoch >= self.early_stopping_rounds) and
                    (epoch + 1 >= self.min_epochs)
                ):
                    print(f"Early stopping at epoch {epoch+1}")
                    print(f"Loading best model from epoch {best_epoch} "
                          f"with Logloss {best_log_loss:.5f}")
                    break

            model.load_state_dict(
                {k: v.to(self.device) for k, v in best_model_state.items()}
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

            # Save OOF
            model.eval()
            with torch.no_grad():
                val_tensor = torch.tensor(X_val).to(self.device)
                val_logits = model(val_tensor)
                val_probs = torch.sigmoid(val_logits)
                oof_preds[val_idx] = val_probs.cpu().numpy().ravel()

                test_tensor = torch.tensor(X_test).to(self.device)
                test_logits = model(test_tensor)
                test_probs = torch.sigmoid(test_logits)
                test_preds += test_probs.cpu().numpy().ravel()

            end = time.time()
            print(f"Best Logloss: {best_log_loss:.5f}")
            print_duration(start, end)

        self.oof_score = log_loss(y, oof_preds)
        print("\n=== CV 結果 ===")
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

    def get_best_fold(self):
        """
        最もスコアの高かったfoldのインデックスとそのスコアを返す。

        Returns
        -------
        best_index: int
            ベストスコアのfoldのインデックス。
        self.fold_scores[best_index] : float
            スコア。
        """
        best_index = int(np.argmax(self.fold_scores))
        return best_index, self.fold_scores[best_index]

    def fit_one_fold(self, tr_df, fold=0):
        """
        指定した1つのfoldのみを用いてモデルを学習する。
        主にOptunaによるハイパーパラメータ探索時に使用。

        Parameters
        ----------
        tr_df : pd.DataFrame
            学習用データ。
        fold : int
            学習に使うfold番号。
        """
        tr_df = tr_df.copy()

        if "weight" in tr_df.columns:
            weights = tr_df["weight"].astype("float32").values
            tr_df = tr_df.drop("weight", axis=1)
        else:
            weights = np.ones(len(tr_df), dtype="float32")

        X = tr_df.drop("target", axis=1).to_numpy().astype(np.float32)
        y = tr_df["target"].to_numpy().astype(np.float32)

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        tr_idx, val_idx = list(skf.split(X, y))[fold]
        start = time.time()

        X_tr, y_tr, w_tr = X[tr_idx], y[tr_idx], weights[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

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
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )

        model = SimpleMLP(
            input_dim=X.shape[1],
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate,
            activation=self.activation
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.t_max,
            eta_min=self.eta_min
        )
        criterion = nn.BCEWithLogitsLoss()

        best_logloss = float("inf")
        best_model_state = None
        best_epoch = 0

        for epoch in range(self.epochs):
            model.train()
            for xb, yb, wb in train_loader:
                xb, yb, wb = (
                    xb.to(self.device),
                    yb.to(self.device),
                    wb.to(self.device)
                )
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
                    xb = xb.to(self.device)
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
                        xb = xb.to(self.device)
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
                (epoch - best_epoch >= self.early_stopping_rounds) and
                (epoch + 1 >= self.min_epochs)
            ):
                print(f"Early stopping at epoch {epoch+1}")
                print(
                    f"Loading best model from epoch {best_epoch} "
                    f"with Logloss {best_logloss:.5f}")
                break

        model.load_state_dict(
            {k: v.to(self.device) for k, v in best_model_state.items()}
        )

        end = time.time()
        print_duration(start, end)
        print(f"Best Logloss: {best_logloss:.5f}")

        self.fold_models.append(MLPFoldModel(
            model,
            X_val,
            y_val,
            0,
            best_rounds=best_epoch
        ))
        self.fold_scores.append(best_logloss)


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

    def __init__(self, model, X_val, y_val, fold_index, best_rounds):
        self.model = model
        self.X_val = X_val
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