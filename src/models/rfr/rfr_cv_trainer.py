from cuml.ensemble import RandomForestRegressor
import numpy as np
import cudf
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
import joblib
import time
from src.utils.print_duration import print_duration


class RFRCVTrainer:
    """
    RFRを使ったGPUでのCVトレーナー。

    Attributes
    ----------
    tr_df : pd.DataFrame
        label付データ
    test_df : pd.DataFrame, default None
        labelなしデータ。CV学習はtest_df必須。
    params : dict
        RFRのパラメータ。
    n_splits : int, default 5
        KFoldの分割数。
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

        if isinstance(tr_df, pd.DataFrame):
            tr_df = cudf.DataFrame.from_pandas(tr_df)

        self.X = tr_df.drop("target", axis=1)
        self.y = tr_df["target"].to_cupy()

        # test
        if test_df is not None:
            if isinstance(test_df, pd.DataFrame):
                self.test = cudf.DataFrame.from_pandas(test_df)
            else:
                self.test = test_df
        else:
            self.test = None

        # fold indices
        skf = KFold(
            n_splits=n_splits, shuffle=True, random_state=self.seed
        )
        self.fold_indices = list(
            skf.split(self.X.to_pandas()))

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

            model = RandomForestRegressor(**self.params)
            model.fit(X_tr, y_tr)

            oof_preds[val_idx] = model.predict(X_val).to_numpy()
            test_preds += model.predict(self.test).to_numpy()

            end = time.time()
            print_duration(start, end)

            score = np.sqrt(
                mse(y_val.to_numpy(), oof_preds[val_idx])
            )
            print(f"Valid RMSE: {score:.5f}")

            self.fold_models.append(RFRFoldModel(
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

        self.oof_score = np.sqrt(mse(self.y.get(), oof_preds))
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

        Rerurn
        ------
        rmse : float
            Score
        """
        start = time.time()
        tr_idx, va_idx = self.fold_indices[fold]

        X_tr, y_tr = self.X.iloc[tr_idx], self.y[tr_idx]
        X_val, y_val = self.X.iloc[va_idx], self.y[va_idx]

        model = RandomForestRegressor(**self.params)
        model.fit(X_tr, y_tr)

        end = time.time()
        print_duration(start, end)

        preds = model.predict(X_val)
        rmse = np.sqrt(mse(y_val.get(), preds.to_numpy()))
        r2 = r2_score(y_val.get(), preds.to_numpy())

        print(f"Valid RMSE: {rmse:.5f}")
        print(f"Valid R^2: {r2:.5f}")

        return rmse


class RFRFoldModel:
    """
    RFRのfold単位モデルを保持するクラス。。

    Attributes
    ----------
    model : cuml.linear_model.RandomForestRegressor
        学習済みのRFRモデル。
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