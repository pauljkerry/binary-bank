from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import shap
import joblib
import time
from src.utils.print_duration import print_duration


class CBCVTrainer:
    """
    CBを使ったCVトレーナー。

    Attributes
    ----------
    params : dict
        CBのパラメータ。
    n_splits : int, default 5
        StratifiedKFoldの分割数。
    early_stopping_rounds : int, default 100
        早期停止ラウンド数。
    seed : int, default 42
        乱数シード。
    """

    def __init__(self, params=None, n_splits=5,
                 early_stopping_rounds=100, seed=42):
        self.params = params or {}
        self.n_splits = n_splits
        self.early_stopping_rounds = early_stopping_rounds
        self.fold_models = []
        self.fold_scores = []
        self.seed = seed
        self.oof_score = None

    def get_default_params(self):
        """
        CB用のデフォルトパラメータを返す。

        Returns
        -------
        default_params : dict
            デフォルトパラメータの辞書。
        """
        default_params = {
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "learning_rate": 0.1,
            "depth": 6,
            "iterations": 20000,
            "min_data_in_leaf": 1,
            "l2_leaf_reg": 3.0,
            "bagging_temperature": 1,
            "random_strength": 10,
            "border_count": 128,
            "grow_policy": "SymmetricTree",
            "random_seed": self.seed,
            "task_type": "GPU",  # or CPU
            "early_stopping_rounds": 100,
            "allow_writing_files": False,
            "verbose": 100
        }
        return default_params

    def fit(self, tr_df, test_df):
        """
        CVを用いてモデルを学習し、OOF予測とtest_dfの平均予測を返す。

        Parameters
        ----------
        tr_df : pd.DataFrame
            学習用データ。
        test_df : pd.DataFrame
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
            weights = tr_df["weight"].astype("float32")
            tr_df = tr_df.drop("weight", axis=1)
        else:
            weights = pd.Series(
                np.ones(len(tr_df), dtype="float32"),
                index=tr_df.index
            )

        X = tr_df.drop("target", axis=1)
        y = tr_df["target"]

        default_params = self.get_default_params()
        self.params = {**default_params, **self.params}

        cat_cols = tr_df.select_dtypes(include="object").columns.to_list()

        oof_preds = np.zeros((len(tr_df), 2))
        test_preds = np.zeros((len(test_df), 2))

        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True,
            random_state=self.seed)

        iteration_list = []

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\nFold {fold + 1}")
            start = time.time()

            X_tr, y_tr, w_tr = (
                X.iloc[tr_idx],
                y.iloc[tr_idx],
                weights.iloc[tr_idx]
            )
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            train_pool = Pool(
                X_tr, y_tr,
                cat_features=cat_cols,
                weight=w_tr
            )
            val_pool = Pool(
                X_val, y_val,
                cat_features=cat_cols
            )

            model = CatBoostClassifier(**self.params)

            model.fit(
                train_pool, eval_set=val_pool, use_best_model=True
            )

            val_preds = model.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = val_preds

            test_preds += model.predict_proba(test_df)[:, 1]

            val_auc = roc_auc_score(y_val, val_preds)
            best_iteration = model.best_iteration_
            print(f"Valid AUC: {val_auc:.5f}")

            end = time.time()
            print_duration(start, end)

            self.fold_models.append(
                CBFoldModel(
                    model, X_val, y_val, fold
                ))
            self.fold_scores.append(val_auc)

            iteration_list.append(best_iteration)

        print("\n=== CV Results ===")
        print(f"Fold scores: {self.fold_scores}")
        print(
            f"Mean: {np.mean(self.fold_scores):.5f}, "
            f"Std: {np.std(self.fold_scores):.5f}"
        )

        self.oof_score = roc_auc_score(y, oof_preds)
        print(f" OOF score: {self.oof_score:.5f}")
        print(f"Avg best iteration: {np.mean(iteration_list)}")
        print(f"Best iterations: \n{iteration_list}")

        test_preds /= self.n_splits

        return oof_preds, test_preds

    def full_train(self, tr_df, test_df, ID, level="l1"):
        """
        訓練データ全体でモデルを学習し、test_dfに対する予測結果をnpy形式で保存する。

        Parameters
        tr_df : pd.DataFrame
            学習用データ。
        test_df : pd.DataFrame
            テスト用データ。
        ID : str
            保存ファイル名に付加する識別子。
        level : str, default "l1"
            保存先のフォルダ名。
        """
        tr_df = tr_df.copy()
        test_df = test_df.copy()

        if "weight" in tr_df.columns:
            weights = tr_df["weight"].astype("float32")
            tr_df = tr_df.drop("weight", axis=1)
        else:
            weights = pd.Series(
                np.ones(len(tr_df), dtype="float32"),
                index=tr_df.index
            )

        X = tr_df.drop("target", axis=1)
        y = tr_df["target"].values

        default_params = self.get_default_params()
        self.params = {**default_params, **self.params}

        cat_cols = tr_df.select_dtypes(include="object").columns.to_list()

        train_pool = Pool(
            X, y, cat_features=cat_cols, weight=weights)

        start = time.time()

        model = CatBoostClassifier(**self.params)

        model.fit(
            train_pool,
            use_best_model=True
        )

        end = time.time()
        print_duration(start, end)

        self.fold_models.append(
            CBFoldModel(
                model, None, None, None
            ))

        test_preds = model.predict_proba(test_df)[:, 1]

        path = f"../artifacts/preds/{level}/test_full_{ID}.npy"
        np.save(path, test_preds)
        print(f"Successfully saved test predictions to {path}")

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
            weights = tr_df["weight"].astype("float32")
            tr_df = tr_df.drop("weight", axis=1)
        else:
            weights = pd.Series(
                np.ones(len(tr_df), dtype="float32"),
                index=tr_df.index
            )

        X = tr_df.drop("target", axis=1)
        y = tr_df["target"]

        default_params = self.get_default_params()
        self.params = {**default_params, **self.params}

        cat_cols = tr_df.select_dtypes(include="object").columns.to_list()

        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True,
            random_state=self.seed)

        tr_idx, val_idx = list(skf.split(X, y))[fold]
        start = time.time()

        X_tr, y_tr, w_tr = (
            X.iloc[tr_idx],
            y.iloc[tr_idx],
            weights.iloc[tr_idx]
        )
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        train_pool = Pool(
            X_tr, y_tr,
            cat_features=cat_cols,
            weight=w_tr
        )
        val_pool = Pool(
            X_val, y_val,
            cat_features=cat_cols
        )

        model = CatBoostClassifier(**self.params)

        model.fit(
            train_pool, eval_set=val_pool, use_best_model=True, 
        )

        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        print(f"Valid AUC: {auc:.5f}")

        self.fold_models.append(CBFoldModel(
            model,
            X_val,
            y_val,
            fold
        ))
        self.fold_scores.append(auc)
        end = time.time()
        print_duration(start, end)


class CBFoldModel:
    """
    CatBoostのfold単位のモデルを保持するクラス。

    Attributes
    ----------
    model : catboost.CatBoostRegressor
        学習済みのCatBoostモデル。
    X_val : pd.DataFrame
        検証用の特徴量データ。
    y_val : pd.Series
        検証用のターゲットラベル。
    fold_index : int
        Foldの番号。
    """

    def __init__(self, model, X_val, y_val, fold_index):
        self.model = model
        self.X_val = X_val
        self.y_val = y_val
        self.fold_index = fold_index

    def shap_plot(self, sample=1000):
        """
        SHAPを用いた特徴量の重要度の可視化を行う。

        Parameters
        ----------
        sample : int, default 1000
            可視化に使用するサンプル数。
        """
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer(self.X_val[:sample])
        shap.summary_plot(shap_values, self.X_val[:sample], max_display=100)

    def save_model(self, path="../artifacts/model/cb_vn.pkl"):
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
        self : CBFoldModel
            読み込んだモデルを保持するインスタンス自身を返す。
        """
        self.model = joblib.load(path)
        return self