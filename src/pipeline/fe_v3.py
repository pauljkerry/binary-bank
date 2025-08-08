import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
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
    - MLP用
    - target encodingを行う
    - defaultをdrop
    """
    # 全データを結合（train + original + test）
    all_data = pd.concat(
        [train_data, test_data], ignore_index=True
    ).drop("target", axis=1, errors="ignore")

    # === 1) カテゴリー変数をTarget Encoding ===
    base_cat_cols = all_data.select_dtypes(
        include=["category", "object"]).columns.difference(["default"])
    cat_cols_te = [col for col in base_cat_cols 
                   if all_data[col].nunique() >= 3]

    te_df = target_encoding(train_data, test_data, cat_cols=cat_cols_te)

    # === 2) カテゴリ変数をOne Hot Encoding ===
    cat_cols_ohe = base_cat_cols.difference(cat_cols_te)
    encoder = OneHotEncoder(
        sparse_output=False, handle_unknown='ignore', drop="first")
    cat_ohe_df = pd.DataFrame(
        encoder.fit_transform(all_data[cat_cols_ohe]),
        columns=encoder.get_feature_names_out(cat_cols_ohe),
        index=all_data.index)

    # === 2) 数値変数を標準化
    num_df = all_data.select_dtypes(include=np.number)
    te_df.index = all_data.index
    merged_df = pd.concat([num_df, te_df], axis=1)

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(merged_df)
    num_scaled_df = pd.DataFrame(
        scaled_array,
        columns=merged_df.columns,
        index=all_data.index
    )
    # === dfを結合 ===
    df_feat = pd.concat([num_scaled_df, cat_ohe_df], axis=1)

    # === データを分割 ===
    tr_df = df_feat.iloc[:len(train_data)].copy()
    test_df = df_feat.iloc[len(train_data):]

    # === targetを追加 ===
    tr_df["target"] = train_data["target"]

    return tr_df, test_df