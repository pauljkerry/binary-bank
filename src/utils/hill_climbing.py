import gc
import os
import re
import numpy as np
import cupy as cp
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score
from src.utils.multiple_auc_scores import multiple_auc_scores


def hill_climbing_auc(
    train_paths: str | list[str],
    test_paths: str | list[str],
    target: str = "target",
    TOL: float = 1e-5,
    USE_NEGATIVE_WGT: float = True
) -> dict:
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
    if isinstance(train_paths, (str, os.PathLike)):
        train_paths = [str(train_paths)]
    else:
        train_paths = [str(p) for p in train_paths]

    if test_paths:
        if isinstance(test_paths, (str, os.PathLike)):
            test_paths = [str(test_paths)]
        else:
            test_paths = [str(p) for p in test_paths]

    oof_df = pl.read_parquet(train_paths)
    test_df = pl.read_parquet(test_paths)

    all_cols = pl.read_parquet(train_paths, n_rows=0).columns
    meta = {
        c
        for c in ("row_id", target)
        if c and c in all_cols
    }
    pat = re.compile(r"^\d+fold(?:-[A-Za-z0-9]+)?$")
    features = [
        c for c in all_cols
        if c not in meta and not pat.fullmatch(c)
    ]

    X = (
        oof_df
        .select(features)
        .to_numpy()
        .astype(np.float32, copy=False)
    )
    y = oof_df.get_column(target)

    test_X = (
        test_df
        .select(features)
        .to_numpy()
        .astype(np.float32, copy=False)
    )

    n_samples, n_models = X.shape

    # 1. 各モデル単体のAUCを計算
    aucs = [roc_auc_score(y, X[:, i]) for i in range(n_models)]
    best_index = np.argmax(aucs)
    best_score = aucs[best_index]

    X_cp = cp.array(X)
    y_cp = cp.array(y)
    best_ensemble = X_cp[:, best_index]

    w_grid = cp.arange(
        -0.50 if USE_NEGATIVE_WGT else 0.01, 0.51, 0.01, dtype=cp.float32
    )

    remaining = set(range(n_models)) - {best_index}
    models = [best_index]
    weights = []
    best_history = [best_score]
    history_rows = [{
        "iteration": 0,
        "model": features[best_index],
        "weight": 1.0,
        "score": round(best_score, 5),
    }]

    old_best = best_score

    print(f"0 We begin with best single model AUC {best_score:0.5f} "
          f"from {features[best_index]}")
    while remaining:
        candidate_score = best_score
        chosen_index = -1
        chosen_weight = 0
        potential_ensemble = None

        # 3. 残りのモデルを1つずつ追加してAUCを計算
        for i in list(remaining):
            new_model = X_cp[:, i]
            mm = (
                best_ensemble[:, None] * (1 - w_grid)[None, :]
                + new_model[:, None] * w_grid[None, :]
            )
            new_scores = multiple_auc_scores(y_cp, mm)
            w_arg = int(np.argmax(new_scores))
            new_best_score = float(new_scores[w_arg])
            if new_best_score > candidate_score:
                candidate_score = new_best_score
                chosen_index = i
                chosen_weight = float(w_grid[w_arg].item())
                potential_ensemble = mm[:, w_arg]
            del new_model, mm, new_scores
            gc.collect()

        # 終了判定
        if (candidate_score - old_best) < TOL or chosen_index < 0:
            print(f'=> We reached tolerance {TOL}')
            break

        print(f"New best score: {candidate_score:.5f}\n"
              f"adding: {features[chosen_index]}\n"
              f"with weight: {chosen_weight:0.3f}\n")

        models.append(chosen_index)
        weights.append(chosen_weight)
        best_history.append(candidate_score)
        best_ensemble = potential_ensemble
        old_best = candidate_score
        best_score = candidate_score
        remaining.remove(chosen_index)

        history_rows.append({
            "iteration": len(models) - 1,
            "model": features[chosen_index],
            "weight": round(chosen_weight, 3),
            "score": round(candidate_score, 5)
        })

    final_weights = np.array([1.0], dtype=np.float32)
    for w in weights:
        final_weights = final_weights * (1.0 - w)
        final_weights = np.concatenate(
            [final_weights, np.array([w], dtype=np.float32)]
        )

    oof_pred = cp.asnumpy(best_ensemble).astype(np.float32, copy=False)
    test_pred = (
        test_X[:, models] @ final_weights.astype(np.float32)
    ).astype(np.float32, copy=False)

    history_df = pd.DataFrame(history_rows)

    # モデル名/重みも並行して返す
    weight_dict = {features[m]: float(w) for m, w in zip(models, final_weights)}

    return {
        "oof_pred": oof_pred,                 # (n_samples,)
        "test_pred": test_pred,               # (n_test,) or None
        "weights": weight_dict,                # 使われた列名
        "history_df": history_df             # 最終のmodel別weight表
    }
