import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import roc_auc_score


def create_objective(oof_list):
    """
    Optunaの目的関数（objective）を生成する関数。

    Parameters
    ----------
    oof_list : list of np.ndarray
        各モデルの予測値（oof）をまとめたリスト

    Returns
    -------
    objective : function
        optunaで使う目的関数
    """
    tr_df3 = pd.read_parquet("../artifacts/features/base/tr_df1.parquet")
    y_true = tr_df3["target"].to_numpy()
    n_models = len(oof_list)

    def objective(trial):
        raw_weights = [trial.suggest_float(f"raw_w{i}", 0.0, 1.0)
                       for i in range(n_models)]
        weight_sum = sum(raw_weights)
        weights = [w / weight_sum for w in raw_weights]  # 正規化

        # 各 oof を「クラス1の確率」に統一
        prob_list = []
        for oof in oof_list:
            if oof.ndim == 1:
                # すでにクラス1の予測（MLPなど）
                prob = oof
            elif oof.shape[1] == 1:
                # 1列2次元（安全対策）
                prob = oof[:, 0]
            else:
                # predict_proba 出力（CatBoostなど）
                prob = oof[:, 1]  # クラス1の確率
            prob_list.append(prob)

        # 加重平均
        y_prob = sum(w * p for w, p in zip(weights, prob_list))

        acc = roc_auc_score(y_true, y_prob)
        return acc

    return objective


def run_optuna_search(
    objective, n_trials=50, direction="minimize", study_name="weight_study",
    storage=None, initial_params: dict = None, sampler=None
):
    """
    Optunaによるハイパーパラメータ探索を実行する関数。

    Parameters
    ----------
    objective : function
        Optunaの目的関数。
    n_trials : int, default 50
        試行回数。
    direction : str, default "minimize"
        Optunaの探索方向。
    study_name : str or None, default "weight_study"
        StudyName。
    storage : str or None, default None
        保存先URL。
    initial_params : dict or None, default None
        初期の試行パラメータ。
    sampler : optuna.samplers.BaseSampler or None, default TPESampler
        使用するSampler。

    Returns
    -------
    study : optuna.Study
        探索結果のStudyオブジェクト。
    """
    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=sampler or optuna.samplers.TPESampler()
    )

    if initial_params is not None:
        study.enqueue_trial(initial_params)

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True
    )

    return study