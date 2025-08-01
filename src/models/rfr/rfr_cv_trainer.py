from cuml.ensemble import RandomForestRegressor
import numpy as np
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
    params : dict
        RFRのパラメータ。
    n_splits : int, default 5
        KFoldの分割数。
    seed : int, default 42
        乱数シード。
    """

    def __init__(self, params=None, n_splits=5, seed=42):
        self.params = params or {}
        self.n_splits = n_splits
        self.fold_models = []
        self.fold_scores = []
        self.seed = seed
        self.oof_score = None

    def get_default_params(self):
        """
        RFR用のデフォルトパラメータを返す。

        Returns
        -------
        default_params : dict
            デフォルトパラメータの辞書。
        """
        default_params = {
            "n_estimators": 100,
            "max_depth": 16,
            "bootstrap": True,
            "random_state": self.seed,
            "n_streams": 1
        }
        return default_params

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
            tr_df = tr_df.drop("weight", axis=1)

        X = tr_df.drop("target", axis=1)
        y = tr_df["target"]

        X_pd = X.to_pandas()

        default_params = self.get_default_params()
        self.params = {**default_params, **self.params}

        oof_preds = np.zeros(len(X))
        test_preds = np.zeros(len(test_df))

        skf = KFold(
            n_splits=self.n_splits, shuffle=True
        )

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_pd)):
            print(f"\nFold {fold + 1}")
            start = time.time()
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            model = RandomForestRegressor(**self.params)
            model.fit(X_tr, y_tr)

            oof_preds[val_idx] = model.predict(X_val).to_numpy()
            test_preds += model.predict(test_df).to_numpy()

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

        print("\n=== CV 結果 ===")
        print(f"Fold scores: {self.fold_scores}")
        print(
            f"Mean: {np.mean(self.fold_scores):.5f}, "
            f"Std: {np.std(self.fold_scores):.5f}"
        )

        self.oof_score = np.sqrt(mse(y.to_numpy(), oof_preds))
        print(f"OOF score: {self.oof_score:.5f}")

        test_preds /= self.n_splits

        return oof_preds, test_preds

    def get_best_fold(self):
        """
        最もスコアの高かったfoldのインデックスを返す。

        Returns
        -------
        best_index: int
            ベストスコアのfoldのインデックス。
        """
        best_index = int(np.argmax(self.fold_scores))
        return best_index

    def fit_one_fold(self, tr_df, fold=0):
        """
        指定した1つのfoldのみを用いてモデルを学習する。
        主にOptunaによるハイパーパラメータ探索時に使用。

        Parameters
        ----------
        tr_df : cudf.DataFrame
            学習用データ。
        fold : int
            学習に使うfold番号。
        """
        tr_df = tr_df.copy()

        if "weight" in tr_df.columns:
            tr_df = tr_df.drop("weight", axis=1)

        X = tr_df.drop("target", axis=1)
        y = tr_df["target"]

        X_pd = X.to_pandas()

        default_params = self.get_default_params()
        self.params = {**default_params, **self.params}

        skf = KFold(
            n_splits=self.n_splits, shuffle=True
        )

        start = time.time()
        tr_idx, va_idx = list(skf.split(X_pd))[fold]

        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_val, y_val = X.iloc[va_idx], y.iloc[va_idx]

        model = RandomForestRegressor(**self.params)
        model.fit(X_tr, y_tr)

        end = time.time()
        print_duration(start, end)

        preds = model.predict(X_val)
        rmse = np.sqrt(mse(y_val.to_numpy(), preds.to_numpy()))
        r2 = r2_score(y_val.to_numpy(), preds.to_numpy())

        print(f"Valid RMSE: {rmse:.5f}")
        print(f"Valid R^2: {r2:.5f}")

        self.fold_models.append(RFRFoldModel(
            model=model,
            X_val=X_val,
            y_val=y_val,
            fold=fold,
        ))
        self.fold_scores.append(rmse)


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