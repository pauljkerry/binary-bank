from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
import numpy as np
import time
from src.utils.print_duration import print_duration


class TabNetCVTrainer:
    def __init__(
        self, n_splits=5, seed=42, max_epochs=100,
        batch_size=1024, early_stopping_rounds=20,
        lr=2e-2, weight_decay=1e-5, use_gpu=True,
        log_interval=10, verbose=10,
        **tabnet_params
    ):
        self.n_splits = n_splits
        self.seed = seed
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.early_stopping_rounds = early_stopping_rounds
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.fold_models = []
        self.fold_scores = []
        self.oof_score = None
        self.log_interval = log_interval
        self.verbose = verbose
        self.tabnet_params = tabnet_params

    def fit(self, tr_df, test_df):
        tr_df = tr_df.copy()
        test_df = test_df.copy()

        if "weight" in tr_df.columns:
            weights = tr_df["weight"].astype("float32").values
            tr_df = tr_df.drop("weight", axis=1)
        else:
            weights = np.ones(len(tr_df), dtype="float32")

        self.cat_cols = tr_df.select_dtypes(include=["object", "category"]).columns.tolist()
        self.num_cols = [col for col in tr_df.columns if col not in self.cat_cols + ["target"]]

        cat_idxs = [tr_df.columns.get_loc(col) for col in self.cat_cols]
        cat_dims = [tr_df[col].nunique() for col in self.cat_cols]

        features = self.num_cols + self.cat_cols

        X = tr_df[features].values.astype(np.float32)
        y = tr_df["target"].values.astype(np.float32)

        X_test = test_df[features].values.astype(np.float32)

        oof_preds = np.zeros(len(X))
        test_preds = np.zeros(len(X_test))

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        epoch_list = []

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\nFold {fold + 1}")
            start = time.time()

            X_train, y_train, w_train = X[tr_idx], y[tr_idx], weights[tr_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            model = TabNetClassifier(
                cat_idxs=cat_idxs,
                cat_dims=cat_dims,
                cat_emb_dim=1,
                n_d=self.hidden_dims[0] if self.hidden_dims else 8,
                n_a=self.hidden_dims[0] if self.hidden_dims else 8,
                n_steps=5,
                gamma=1.5,
                n_independent=2,
                n_shared=2,
                momentum=0.3,
                lambda_sparse=1e-3,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=self.lr),
                scheduler_params={
                    "T_max": self.t_max,
                    "eta_min": self.eta_min,
                    "verbose": False
                },
                scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
                mask_type='entmax',
                verbose=self.log_interval,
                seed=self.seed,
                device_name="cuda" if torch.cuda.is_available() else "cpu"
            )

            model.fit(
                X_train=X_train,
                y_train=y_train.reshape(-1, 1),
                eval_set=[(X_val, y_val.reshape(-1, 1))],
                eval_metric=["logloss"],
                max_epochs=self.epochs,
                patience=self.early_stopping_rounds,
                batch_size=self.batch_size,
                virtual_batch_size=128,
                num_workers=0,
                drop_last=False,
                weights=w_train
            )

            val_preds = model.predict_proba(X_val)[:, 1]
            val_logloss = roc_auc_score(y_val, val_preds)
            print(f"Best Logloss: {val_logloss:.5f}")

            self.fold_models.append(model)
            self.fold_scores.append(val_logloss)
            epoch_list.append(len(model.history["loss"]))

            oof_preds[val_idx] = val_preds
            test_preds += model.predict_proba(X_test)[:, 1]

            end = time.time()
            print_duration(start, end)

        self.oof_score = roc_auc_score(y, oof_preds)
        print("\n=== CV Results ===")
        print(f"Fold scores: {self.fold_scores}")
        print(f"Mean: {np.mean(self.fold_scores):.5f}, Std: {np.std(self.fold_scores):.5f}")
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

        self.cat_cols = tr_df.select_dtypes(
            include=["object", "category"]).columns.tolist()
        self.num_cols = [col for col in tr_df.columns
                         if col not in self.cat_cols + ["target"]]
        categorical_cardinalities = {
            col: tr_df[col].nunique()
            for col in self.cat_cols
        }

        X_num = tr_df[self.num_cols].to_numpy().astype(np.float32)
        X_cat = tr_df[self.cat_cols].to_numpy().astype(np.int64)
        y = tr_df["target"].to_numpy().astype(np.float32)

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        tr_idx, val_idx = list(skf.split(X_num, y))[fold]
        start = time.time()

        X_tr_num, X_tr_cat, y_tr, w_tr = (
            X_num[tr_idx], X_cat[tr_idx], y[tr_idx], weights[tr_idx])
        X_val_num, X_val_cat, y_val = X_num[val_idx], X_cat[val_idx], y[val_idx]

        # Dataloaders
        train_dataset = TensorDataset(
            torch.tensor(X_tr_num),
            torch.tensor(X_tr_cat),
            torch.tensor(y_tr),
            torch.tensor(w_tr)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val_num),
            torch.tensor(X_val_cat),
            torch.tensor(y_val)
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )

        model = SimpleMLP(
            input_dim=X_num.shape[1],
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            categorical_cardinalities=categorical_cardinalities
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
            for xb_num, xb_cat, yb, wb in train_loader:
                xb_num = xb_num.to(self.device)
                yb = yb.to(self.device)
                wb = wb.to(self.device)
                x_cat_dict = {
                    col: xb_cat[:, i].long().to(self.device)
                    for i, col in enumerate(self.cat_cols)
                }

                preds = model(xb_num, x_cat_dict)
                loss = criterion(preds, yb)
                loss = (loss * wb).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            preds = []
            with torch.no_grad():
                for xb_num, xb_cat, yb in val_loader:
                    xb_num = xb_num.to(self.device)
                    x_cat_dict = {
                        col: xb_cat[:, i].long().to(self.device)
                        for i, col in enumerate(self.cat_cols)
                    }
                    pred_logits = model(xb_num, x_cat_dict)
                    pred_probs = torch.sigmoid(pred_logits).cpu().numpy()
                    preds.append(pred_probs)
            val_pred = np.concatenate(preds)
            val_logloss = roc_auc_score(y_val, val_pred)
            scheduler.step()

            if (epoch + 1) % 1 == 0 or epoch == 0:
                model.eval()

                train_preds = []
                train_targets = []
                with torch.no_grad():
                    for xb_num, xb_cat, yb, wb in train_loader:
                        xb_num = xb_num.to(self.device)
                        x_cat_dict = {
                            col: xb_cat[:, i].long().to(self.device)
                            for i, col in enumerate(self.cat_cols)
                        }
                        pred_logits = model(xb_num, x_cat_dict)
                        pred_probs = torch.sigmoid(pred_logits).cpu().numpy()
                        train_preds.append(pred_probs)
                        train_targets.append(yb.numpy())
                train_preds = np.concatenate(train_preds)
                train_targets = np.concatenate(train_targets)
                train_roc_auc_score = roc_auc_score(train_targets, train_preds)

                print(
                    f"Epoch {epoch+1}: "
                    f"Train Logloss = {train_roc_auc_score:.5f}, "
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
            X_val_num,
            X_val_cat,
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
    X_val_num : pd.DataFrame
        検証用の数値特徴量データ。
    X_val_cat : pd.DataFrame
        検証用のカテゴリ特徴量データ。
    y_val : pd.Series
        検証用のターゲットラベル。
    fold_index : int
        foldの番号。
    best_rounds : int
        最良スコア時のエポック数
    """

    def __init__(
        self, model, X_val_num, X_val_cat, y_val, fold_index, best_rounds
    ):
        self.model = model
        self.X_val_num = X_val_num
        self.X_val_cat = X_val_cat
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