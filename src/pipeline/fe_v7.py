import pandas as pd
import numpy as np
from src.utils.target_encoding import target_encoding


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

    # === 1) カテゴリー変数をTarget Encoding ===
    cat_cols = all_data.select_dtypes(
        include=["category", "object"]).columns

    te_df = target_encoding(train_data, test_data, cat_cols=cat_cols).reset_index(drop=True)

    # === dfを結合 ===
    num_df = all_data.select_dtypes(include=np.number).reset_index(drop=True)
    df_feat = pd.concat([num_df, te_df], axis=1)

    # === データを分割 ===
    tr_df = df_feat.iloc[:len(train_data)].copy()
    test_df = df_feat.iloc[len(train_data):]

    # === targetを追加 ===
    tr_df["target"] = train_data["target"]

    return tr_df, test_df