import xgboost as xgb
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
from src.utils.print_duration import print_duration


class XGBCVTrainer:
    """
    XGBを使ったCVトレーナー。

    Attributes
    ----------
    params : dict
        XGBのパラメータ。
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
        XGB用のデフォルトパラメータを返す。

        Returns
        -------
        default_params : dict
            デフォルトパラメータの辞書。
        """
        default_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "learning_rate": 0.1,
            "max_depth": 7,
            "min_child_weight": 10.0,
            "gamma": 0,
            "colsample_bytree": 0.8,
            "subsample": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "verbosity": 0,
            "tree_method": "gpu_hist",
            "random_state": self.seed,
            "max_bin": 512,
            "grow_policy": "depthwise",
            "single_precision_histogram": True,
            "predictor": "gpu_predictor"
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

        cat_cols = tr_df.select_dtypes(include="object").columns
        tr_df[cat_cols] = tr_df[cat_cols].astype("category")
        test_df[cat_cols] = test_df[cat_cols].astype("category")

        dtest = xgb.DMatrix(test_df, enable_categorical=True)

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

        oof_preds = np.zeros(len(X))
        test_preds = np.zeros(len(test_df))

        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True,
            random_state=self.seed
        )

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

            dtrain = xgb.DMatrix(
                X_tr, label=y_tr,
                weight=w_tr, enable_categorical=True
            )
            dvalid = xgb.DMatrix(
                X_val, label=y_val,
                enable_categorical=True
            )
            evals_result = {}

            model = xgb.train(
                self.params,
                dtrain,
                num_boost_round=20000,
                evals=[(dtrain, "train"), (dvalid, "eval")],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=100,
                evals_result=evals_result
            )

            # oof
            oof_preds[val_idx] = model.predict(dvalid)
            test_preds += model.predict(dtest)

            end = time.time()
            print_duration(start, end)

            best_iter = model.best_iteration
            train_score = evals_result["train"]["logloss"][best_iter]
            eval_score = evals_result["eval"]["logloss"][best_iter]
            print(f"Train Logloss: {train_score:.5f}")
            print(f"Valid Logloss: {eval_score:.5f}")

            self.fold_models.append(
                XGBFoldModel(model, X_val, y_val, fold))
            self.fold_scores.append(eval_score)

            iteration_list.append(best_iter)

        print("\n=== CV 結果 ===")
        print(f"Fold scores: {self.fold_scores}")
        print(
            f"Mean: {np.mean(self.fold_scores):.5f}, "
            f"Std: {np.std(self.fold_scores):.5f}"
        )

        self.oof_score = log_loss(y, oof_preds)
        print(f"OOF score: {self.oof_score:.5f}")
        print(f"Avg best iteration: {np.mean(iteration_list)}")
        print(f"Best iterations: \n{iteration_list}")

        test_preds /= self.n_splits

        return oof_preds, test_preds

    def full_train(self, tr_df, test_df, iterations, ID, level="l1"):
        """
        訓練データ全体でモデルを学習し、test_dfに対する予測結果をnpy形式で保存する。

        Parameters
        tr_df : pd.DataFrame
            学習用データ。
        test_df : pd.DataFrame
            テスト用データ。
        iterations : int
            学習の繰り返し回数。
        ID : str
            保存ファイル名に付加する識別子。
        level : str, default "l1"
            保存先のフォルダ名。
        """
        tr_df = tr_df.copy()
        test_df = test_df.copy()

        cat_cols = tr_df.select_dtypes(include="object").columns
        tr_df[cat_cols] = tr_df[cat_cols].astype("category")
        test_df[cat_cols] = test_df[cat_cols].astype("category")

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

        dtrain = xgb.DMatrix(
            X, label=y,
            weight=weights, enable_categorical=True
        )
        dtest = xgb.DMatrix(test_df, enable_categorical=True)

        start = time.time()

        model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=int(iterations*1.25),
            evals=[]
        )

        end = time.time()
        print_duration(start, end)

        test_preds = model.predict(dtest)

        path = f"../artifacts/preds/{level}/test_full_{ID}.npy"
        np.save(path, test_preds)
        print(f"Successfully saved test predictions to {path}")

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
        tr_df : pd.DataFrame
            学習用データ。
        fold : int
            学習に使うfold番号。
        """
        tr_df = tr_df.copy()
        cat_cols = tr_df.select_dtypes(include="object").columns
        tr_df[cat_cols] = tr_df[cat_cols].astype("category")

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

        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=42
        )

        tr_idx, val_idx = list(skf.split(X, y))[fold]
        start = time.time()

        X_tr, y_tr, w_tr = X.iloc[tr_idx], y.iloc[tr_idx], weights.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr,
                             weight=w_tr, enable_categorical=True)
        dvalid = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

        evals_result = {}

        model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=20000,
            evals=[(dtrain, "train"), (dvalid, "eval")],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=100,
            evals_result=evals_result
        )

        end = time.time()
        print_duration(start, end)

        best_iter = model.best_iteration
        train_score = evals_result["train"]["logloss"][best_iter]
        eval_score = evals_result["eval"]["logloss"][best_iter]
        print(f"Train Logloss: {train_score:.5f}")
        print(f"Valid Logloss: {eval_score:.5f}")

        self.fold_models.append(XGBFoldModel(model, X_val, y_val, fold))
        self.fold_scores.append(eval_score)


class XGBFoldModel:
    """
    XGBのfold単位のモデルを保持するクラス。

    Attributes
    ----------
    model : xgb.Booster
        学習済みのXGBoostモデル。
    X_val : pd.DataFrame
        検証用の特徴量データ。
    y_val : pd.Series
        検証用のターゲットラベル。
    fold_index : int
        Foldの番号。
    """

    def __init__(self, model, X_val, y_val, fold_index):
        self.model = model
        self.X_valid = X_val
        self.y_valid = y_val
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
        shap_values = explainer(self.X_valid[:sample])
        shap.summary_plot(shap_values, self.X_valid[:sample])

    def plot_gain_importance(self):
        """
        特徴量のTotalGainに基づく重要度を棒グラフで可視化する。
        """
        importances = self.model.get_score(importance_type="total_gain")

        total_gain = sum(importances.values())
        importance_ratios = [
            np.round((v/total_gain)*100, 2)
            for k, v in importances.items()
        ]
        df = pd.DataFrame({
            "Feature": [k for k in importances.keys()],
            "ImportanceRatio": importance_ratios,
        }).sort_values("ImportanceRatio", ascending=False)

        fig, ax = plt.subplots(figsize=(12, max(4, len(df)*0.4)))
        sns.barplot(
            data=df,
            y="Feature",
            x="ImportanceRatio",
            orient="h",
            palette="viridis",
            hue="Feature",
            ax=ax
        )
        for container in ax.containers:
            labels = ax.bar_label(container)
            for label in labels:
                label.set_fontsize(20)
        plt.title("Feature Importance", fontsize=32)
        plt.xlabel("Importance", fontsize=28)
        plt.ylabel("Feature", fontsize=28)
        ax.tick_params(axis="x", labelsize=20)
        ax.tick_params(axis="y", labelsize=20)
        plt.tight_layout()
        plt.show()

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