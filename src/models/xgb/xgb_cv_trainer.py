import xgboost as xgb
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
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

    def __init__(self, tr_df, test_df=None, params=None, n_splits=5,
                 early_stopping_rounds=100, num_boost_round=20000, seed=42):
        self.params = params
        self.n_splits = n_splits
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        self.fold_models = []
        self.fold_scores = []
        self.seed = seed
        self.oof_score = None

        # object → category
        cat_cols = tr_df.select_dtypes(include="object").columns
        tr_df[cat_cols] = tr_df[cat_cols].astype("category")

        # 重み
        if "weight" in tr_df.columns:
            self.weights = tr_df["weight"].astype("float32").to_numpy()
            tr_df = tr_df.drop("weight", axis=1)
        else:
            self.weights = np.ones(len(tr_df), dtype="float32")

        # target
        self.X = tr_df.drop("target", axis=1)
        self.y = tr_df["target"].to_numpy()

        # test
        if test_df is not None:
            test_df[cat_cols] = test_df[cat_cols].astype("category")
            self.dtest = xgb.DMatrix(
                test_df, enable_categorical=True)
        else:
            self.dtest = None

        # fold indices
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        self.fold_indices = list(skf.split(self.X, self.y))

        self.default_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "learning_rate": 0.1,
            "max_depth": 7,
            "min_child_weight": 10.0,
            "gamma": 0,
            "colsample_bytree": 0.8,
            "subsample": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "verbosity": 0,
            "tree_method": "hist",
            "device": "cuda",
            "random_state": self.seed,
            "max_bin": 512,
            "grow_policy": "depthwise",
            "single_precision_histogram": True,
            "predictor": "gpu_predictor"
        }

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
        if self.dtest is None:
            raise ValueError("test_df not provided for XGBCVTrainer.")

        self.params = {**self.default_params, **(self.params or {})}
        oof_preds = np.zeros(len(self.X))
        test_preds = np.zeros(self.test_dmat.num_row())

        iteration_list = []

        for fold, (tr_idx, val_idx) in enumerate(self.fold_indices):
            print(f"\nFold {fold + 1}")
            start = time.time()

            X_tr, y_tr, w_tr = (
                self.X.iloc[tr_idx],
                self.y[tr_idx],
                self.weights[tr_idx]
            )
            X_val, y_val = self.X.iloc[val_idx], self.y[val_idx]

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
                num_boost_round=self.num_boost_round,
                evals=[(dtrain, "train"), (dvalid, "eval")],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=100,
                evals_result=evals_result
            )

            # oof
            oof_preds[val_idx] = model.predict(dvalid, iteration_range=(0, model.best_iteration+1))
            test_preds += model.predict(self.dtest, iteration_range=(0, model.best_iteration+1))

            end = time.time()
            print_duration(start, end)

            best_iter = model.best_iteration
            train_score = evals_result["train"]["auc"][best_iter]
            eval_score = evals_result["eval"]["auc"][best_iter]
            print(f"Train AUC: {train_score:.5f}")
            print(f"Valid AUC: {eval_score:.5f}")

            self.fold_models.append(
                XGBFoldModel(model, X_val, y_val, fold))
            self.fold_scores.append(eval_score)

            iteration_list.append(best_iter)

        print("\n=== CV Results ===")
        print(f"Fold scores: {self.fold_scores}")
        print(
            f"Mean: {np.mean(self.fold_scores):.5f}, "
            f"Std: {np.std(self.fold_scores):.5f}"
        )

        self.oof_score = roc_auc_score(self.y, oof_preds)
        print(f"OOF score: {self.oof_score:.5f}")
        print(f"Avg best iteration: {np.mean(iteration_list)}")
        print(f"Best iterations: \n{iteration_list}")

        test_preds /= self.n_splits

        return oof_preds, test_preds

    def full_train(self, tr_df, test_df, iterations, ID, level="l1"):
        """
        訓練データ全体でモデルを学習し、test_dfに対する予測結果をnpy形式で保存する。

        Parameters
        ----------
        iterations : int
            学習の繰り返し回数。
        ID : str
            保存ファイル名に付加する識別子。
        level : str, default "l1"
            保存先のフォルダ名。
        """
        if self.dtest is None:
            raise ValueError("test_df not provided for XGBCVTrainer.")

        self.params = {**self.default_params, **(self.params or {})}
        dtrain = xgb.DMatrix(
            self.X, label=self.y,
            weight=self.weights, enable_categorical=True
        )

        start = time.time()

        model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=int(iterations*1.25),
            evals=[]
        )

        end = time.time()
        print_duration(start, end)

        test_preds = model.predict(self.dtest)

        path = f"../artifacts/preds/{level}/test_full_{ID}.npy"
        np.save(path, test_preds)
        print(f"Successfully saved test predictions to {path}")

    def fit_one_fold(self, fold=0):
        """
        指定した1つのfoldのみを用いてモデルを学習する。
        主にOptunaによるハイパーパラメータ探索時に使用。

        Parameters
        ----------
        fold : int
            学習に使うfold番号。
        """
        self.params = {**self.default_params, **(self.params or {})}
        tr_idx, val_idx = self.fold_indices[fold]
        start = time.time()

        X_tr, y_tr, w_tr = self.X.iloc[tr_idx], self.y[tr_idx], self.weights[tr_idx]
        X_val, y_val = self.X.iloc[val_idx], self.y[val_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr,
                             weight=w_tr, enable_categorical=True)
        dvalid = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

        evals_result = {}

        model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=[(dtrain, "train"), (dvalid, "eval")],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=100,
            evals_result=evals_result
        )

        end = time.time()
        print_duration(start, end)

        best_iter = model.best_iteration
        train_score = evals_result["train"]["auc"][best_iter]
        eval_score = evals_result["eval"]["auc"][best_iter]
        print(f"Train AUC: {train_score:.5f}")
        print(f"Valid AUC: {eval_score:.5f}")

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
        sample_X = self.X_valid[:sample].copy()
        for col in sample_X.select_dtypes(include="category").columns:
            sample_X[col] = sample_X[col].cat.codes

        explainer = shap.TreeExplainer(
            self.model, feature_perturbation='interventional')
        shap_values = explainer.shap_values(sample_X)
        shap.summary_plot(shap_values, sample_X)

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