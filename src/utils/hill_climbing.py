import numpy as np
import cupy as cp
import pandas as pd
import gc
from sklearn.metrics import roc_auc_score
from src.utils.multiple_auc_scores import multiple_auc_scores
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def hill_climbing_auc(
    oof_array,
    y_true,
    test_array=None,
    files=None,
    TOL=1e-5,
    USE_NEGATIVE_WGT=True
):
    """
    AUCでHill Climbingを行う関数

    Parameters
    ----------
    oof_array : np.ndarray
        (n_samples, n_models)のNumPy配列
    y_true : np.ndarray
        (n_samples,)の正解ラベル
    test_array : np.ndarray, default None
    files : list[str or int], default None
        OOFに対応する名前。
        Noneの場合は0からの連番
    TOL : int, default 1e-5
        Hill Climbingでモデルの追加に必要な最小の更新スコア
    USE_NEGATIVE_WGT : bool, default True
        負の重みを使うかどうか

    Retruns
    -------
    ens_pred : np.ndarray or None
        ensembleの予測値
        test_arrayがNoneのときは返り値なし
    """
    n_samples, n_models = oof_array.shape
    if files is None:
        files = list(range(n_models))

    # 1. 各モデル単体のAUCを計算
    aucs = [roc_auc_score(y_true, oof_array[:, i]) for i in range(n_models)]
    best_index = np.argmax(aucs)

    # 2. 初期ベストモデル
    best_models = [best_index]
    best_score = aucs[best_index]
    old_best_score = best_score

    oof_array2 = cp.array(oof_array)
    truth = cp.array(y_true)
    best_ensemble = oof_array2[:, best_index]

    start = -0.50
    if not USE_NEGATIVE_WGT:
        start = 0.01
    ww = cp.arange(start, 0.51, 0.01)
    nn = len(ww)

    models = [best_index]
    weights = []
    best_history = [best_score]

    remaining = set(range(n_models)) - set(best_models)

    print(f"0 We begin with best single model AUC {best_score:0.5f} "
          f"from {files[best_index]}")
    while remaining:
        candidate_score = best_score
        best_index = -1
        best_weight = 0

        # 3. 残りのモデルを1つずつ追加してAUCを計算
        for i in remaining:
            if i in models:
                continue
            new_model = oof_array2[:, i]
            m1 = cp.repeat(best_ensemble[:, cp.newaxis], nn, axis=1) * (1-ww)
            m2 = cp.repeat(new_model[:, cp.newaxis], nn, axis=1) * ww
            mm = m1 + m2
            new_scores = multiple_auc_scores(truth, mm)
            new_best_score = cp.max(new_scores)
            if new_best_score > candidate_score:
                candidate_score = new_best_score
                best_index = i
                best_wgt_idx = np.argmax(new_scores).item()
                best_weight = ww[best_wgt_idx].item()
                potential_ensemble = mm[:, best_wgt_idx]
            del new_model, m1, m2, mm, new_scores, new_best_score
            gc.collect()

        # 終了判定
        if (candidate_score - old_best_score) < TOL:
            print(f'=> We reached tolerance {TOL}')
            break

        print(f"New best score: {candidate_score:.5f}\n"
              f"adding: {files[best_index]}\n"
              f"with weight: {best_weight:0.3f}\n")
        models.append(best_index)
        weights.append(best_weight)
        best_history.append(candidate_score)
        best_ensemble = potential_ensemble
        old_best_score = candidate_score
        remaining.remove(best_index)

    wgt = np.array([1])
    for w in weights:
        wgt = wgt*(1-w)
        wgt = np.concatenate([wgt, np.array([w])])

    rows = []
    t = 0
    for m, w, s in zip(models, wgt, best_history):
        name = files[m]
        dd = {}
        dd['weight'] = w
        dd['model'] = name
        dd["score"] = np.round(s, 5)
        rows.append(dd)
        t += float(f'{w:.3f}')

    # DISPLAY WEIGHT PER MODEL
    df = pd.DataFrame(rows)
    wgt_df = df[["weight", "model"]].groupby('model').agg('sum').reset_index()
    wgt_df = wgt_df.sort_values('weight', ascending=False).reset_index(drop=True)
    print(wgt_df)

    plt.figure(figsize=(8, 5))
    plt.plot(df.index, df["score"], marker="o", label="AUC")
    plt.xlabel("Iteration")
    plt.ylabel("AUC")
    plt.title("AUC Progression during Hill Climbing")
    plt.grid(True)
    plt.legend()
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.5f'))

    plt.show()

    if test_array is not None:
        ens_preds = np.zeros(len(test_array))
        for i, idx in enumerate(models):
            ens_preds += test_array[:, idx] * wgt[i]
        return ens_preds