import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def target_encoding(
    tr_df, test_df, target_col="target", cat_cols=None,
    n_splits=5, seed=42
):
    """
    Target Encodingを行う関数

    Parameters
    ----------
    tr_df : pd.DataFrame
        Label付きデータ
    test_df : pd.DataFrame
        Labelなしデータ
    target_col : str, default "target"
        目的変数の列名
    cat_cols : list, default None
        質的変数名のリスト。
        Noneの場合はすべての質的変数を用いる
    n_splits : int, default 5
        KFoldの分割数
    seed : int, default 42
        乱数シード

    Returns
    -------
    te_df : pd.DataFrame
        Target EncodingしたものをまとめたDF
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    te_tr = pd.DataFrame(index=tr_df.index)
    te_test = pd.DataFrame(index=test_df.index)

    if cat_cols is None:
        cat_cols = tr_df.select_dtypes(
            include=["category", "object"]).columns.difference([target_col])

    for col in cat_cols:
        oof = np.zeros(len(tr_df))
        test_vals = np.zeros(len(test_df))

        for tr_idx, val_idx in skf.split(tr_df, tr_df[target_col]):
            tr_fold = tr_df.iloc[tr_idx]
            val_fold = tr_df.iloc[val_idx]

            means = tr_fold.groupby(col)[target_col].mean()
            oof[val_idx] = val_fold[col].map(means)

            test_vals += test_df[col].map(means).fillna(means.mean()).to_numpy()

        # 平均（テストは各foldの平均→kで割る）
        te_tr[f"{col}_te"] = oof
        te_test[f"{col}_te"] = test_vals / n_splits

    te_df = pd.concat([te_tr, te_test], axis=0)

    return te_df