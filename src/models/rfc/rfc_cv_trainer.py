from cuml.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import joblib
import time
from src.utils.print_duration import print_duration


class RFCCVTrainer:
    """
    RFCを使ったGPUでのCVトレーナー。

    Attributes
    ----------
    tr_df : pd.DataFrame
        label付データ
    test_df : pd.DataFrame, default None
        labelなしデータ。CV学習はtest_df必須。
    params : dict
        RFCのパラメータ。
    n_splits : int, default 5
        StratifiedKFoldの分割数。
    seed : int, default 42
        乱数シード。
    """

    def __init__(self, tr_df, test_df=None, params=None, n_splits=5, seed=42):
        self.params = params or {}
        self.n_splits = n_splits
        self.fold_models = []
        self.fold_scores = []
        self.seed = seed
        self.oof_score = None

        if "weight" in tr_df.columns:
            tr_df = tr_df.drop("weight", axis=1)

        self.X = tr_df.drop("target", axis=1)
        self.y = tr_df["target"].to_cupy()

        # test
        if test_df is not None:
            self.test = test_df
        else:
            self.test = None

        # fold indices
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        self.fold_indices = list(
            skf.split(self.X.to_pandas(), self.y.get()))

        self.default_params = {
            "n_estimators": 100,
            "max_depth": 16,
            "bootstrap": True,
            "random_state": self.seed,
            "n_streams": 1
        }
        self.params = {**self.default_params, **self.params}

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
            raise ValueError("test_df not provided for RFCCVTrainer.")

        oof_preds = np.zeros(len(self.X))
        test_preds = np.zeros(len(self.test))

        for fold, (tr_idx, val_idx) in enumerate(self.fold_indices):
            print(f"\nFold {fold + 1}")
            start = time.time()
            X_tr, y_tr = self.X.iloc[tr_idx], self.y[tr_idx]
            X_val, y_val = self.X.iloc[val_idx], self.y[val_idx]

            model = RandomForestClassifier(**self.params)
            model.fit(X_tr, y_tr)

            oof_preds[val_idx] = model.predict_proba(X_val).to_numpy()[:, 1]
            test_preds += model.predict_proba(self.test).to_numpy()[:, 1]

            end = time.time()
            print_duration(start, end)

            score = roc_auc_score(y_val.get(), oof_preds[val_idx])

            print(f"Valid AUC: {score:.5f}")

            self.fold_models.append(RFCFoldModel(
                model=model,
                X_val=X_val,
                y_val=y_val,
                fold=fold,
            ))
            self.fold_scores.append(score)

        print("\n=== CV Results ===")
        print(f"Fold scores: {self.fold_scores}")
        print(
            f"Mean: {np.mean(self.fold_scores):.5f}, "
            f"Std: {np.std(self.fold_scores):.5f}"
        )

        self.oof_score = roc_auc_score(self.y.get(), oof_preds)
        print(f"OOF score: {self.oof_score:.5f}")

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

        Return
        ------
        score : float
            Score
        """
        start = time.time()
        tr_idx, va_idx = self.fold_indices[fold]

        X_tr, y_tr = self.X.iloc[tr_idx], self.y[tr_idx]
        X_val, y_val = self.X.iloc[va_idx], self.y[va_idx]

        model = RandomForestClassifier(**self.params)
        model.fit(X_tr, y_tr)

        end = time.time()
        print_duration(start, end)

        preds = model.predict_proba(X_val).to_numpy()[:, 1]
        auc = roc_auc_score(y_val.get(), preds)

        print(f"Valid AUC: {auc:.5f}")

        return auc


class RFCFoldModel:
    """
    RFCのfold単位モデルを保持するクラス。。

    Attributes
    ----------
    model : cuml.linear_model.RandomForestClassifier
        学習済みのRFCモデル。
    X_val : cudf.DataFrame
        検証用の特徴量データ。
    y_val : cudf.Series
        検証用のターゲットラベル。
    fold_index : int
        foldの番号。
    """

    def __init__(self, model, X_val, y_val, fold):
        self.model = model
        self.X_val = X_val
        self.y_val = y_val
        self.fold = fold

    def save_model(self, path="../artifacts/model/logreg_vn.pkl"):
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
        self : LogRegFoldModel
            読み込んだモデルを保持するインスタンス自身を返す。
        """
        self.model = joblib.load(path)
        return self