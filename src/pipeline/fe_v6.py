import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def feature_engineering(train_data, test_data):
    """
    特徴量エンジニアリングを行う関数

    Parameters
    ----------
    train_data : pd.DataFrame
        前処理済みの学習用データ
    test_data : pd.DataFrame
        前処理済みのテスト用データ

    Returns
    -------
    tr_df : pd.DataFrame
        特徴量エンジニアリング済みの学習用データ
    test_df : pd.DataFrame
        特徴エンジニアリング済みのテスト用データ

    Notes
    -----
    - GBDT用
    - 特徴量エンジニアリングはせず
    """
    # 全データを結合（train + original + test）
    all_data = pd.concat(
        [train_data, test_data], ignore_index=True
    ).drop("target", axis=1, errors="ignore")

    # === 1) カテゴリー変数をlabel encoding ===
    cat_cols = all_data.select_dtypes(include=["category", "object"]).columns
    le_df = pd.DataFrame(index=all_data.index)

    for c in cat_cols:
        le = LabelEncoder()
        le_df[c] = le.fit_transform(all_data[c])
    le_df = le_df.astype("str")

    # === 2) 数値変数を分位点でBin分割 ===
    bins_df = pd.DataFrame(index=all_data.index)
    bins = 100

    bins_df["age"] = all_data["age"]
    bins_df["day"] = all_data["day"]

    balance_mask = (all_data["balance"] == 0)
    campaign_mask = (all_data["campaign"] == 0)
    pdays_mask = (all_data["pdays"] == -1)
    previous_mask = (all_data["previous"] == 0)
    duration_mask = (all_data["duration"] == 0)

    bins_df.loc[~balance_mask, "balance"] = pd.qcut(
        all_data.loc[~balance_mask, "balance"],
        q=bins, duplicates="drop", labels=False)
    bins_df.loc[balance_mask, "balance"] = -1

    bins_df.loc[~campaign_mask, "campaign"] = pd.qcut(
        all_data.loc[~campaign_mask, "campaign"],
        q=bins, duplicates="drop", labels=False)
    bins_df.loc[campaign_mask, "campaign"] = -1

    bins_df.loc[~pdays_mask, "pdays"] = pd.qcut(
        all_data.loc[~pdays_mask, "pdays"],
        q=bins, duplicates="drop", labels=False)
    bins_df.loc[pdays_mask, "pdays"] = -1

    bins_df.loc[~previous_mask, "previous"] = pd.qcut(
        all_data.loc[~previous_mask, "previous"],
        q=bins, duplicates="drop", labels=False)
    bins_df.loc[previous_mask, "previous"] = -1

    bins_df.loc[~duration_mask, "duration"] = pd.qcut(
        all_data.loc[~duration_mask, "duration"],
        q=bins, duplicates="drop", labels=False)
    bins_df.loc[duration_mask, "duration"] = -1

    bins_df = bins_df.astype("str")

    # === dfを結合 ===
    df_feat = pd.concat([bins_df, le_df], axis=1)

    # === データを分割 ===
    tr_df = df_feat.iloc[:len(train_data)].copy()
    test_df = df_feat.iloc[len(train_data):]

    # === targetを追加 ===
    tr_df["target"] = train_data["target"]

    return tr_df, test_df