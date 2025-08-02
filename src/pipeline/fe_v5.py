import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler
from itertools import combinations


def feature_engineering(train_data, test_data):
    """
    特徴量エンジニアリングを行う関数

    Parameters
    ----------
    train_data : pl.DataFrame
        前処理済みの学習用データ
    test_data : pl.DataFrame
        前処理済みのテスト用データ

    Returns
    -------
    tr_df : pl.DataFrame
        特徴量エンジニアリング済みの学習用データ
    test_df : pl.DataFrame
        特徴エンジニアリング済みのテスト用データ

    Notes
    -----
    - LogReg用
    - 3変数交互作用を追加
    """
    # 全データを結合（train + test）
    all_data = pl.concat([train_data, test_data], how="vertical")

    # === 1) カテゴリー変数をOne Hot Encoding ===
    cat_cols = [col for col in all_data.columns if all_data[col].dtype == pl.Utf8]
    ohe_df = all_data.select(cat_cols).to_dummies()

    # === 2) 数値変数を標準化
    num_cols = [col for col in all_data.columns if all_data[col].dtype in [pl.Float64, pl.Int64] and col != "target"]
    num_df = all_data.select(num_cols)
    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(num_df.to_numpy())
    num_scaled_df = pl.DataFrame(num_scaled, schema=num_cols)

    # === 2) 交互作用を追加
    merged_df = pl.concat([ohe_df, num_scaled_df], how="horizontal")
    inter_2 = []
    # inter_3 = []
    colnames = merged_df.columns

    for col1, col2 in combinations(colnames, 2):
        inter_2.append((merged_df[col1] * merged_df[col2]).alias(f"{col1}_{col2}"))

    """
    for col1, col2, col3 in combinations(colnames, 3):
        inter_3.append((merged_df[col1] * merged_df[col2] * merged_df[col3]).alias(f"{col1}_{col2}_{col3}"))
    """
    inter_df = pl.DataFrame(inter_2)

    # === dfを結合 ===
    feat_df = pl.concat([num_scaled_df, ohe_df, inter_df], how="horizontal")

    # === データを分割 ===
    n_train = train_data.height
    tr_df = feat_df.slice(0, n_train).with_columns(train_data["target"])
    test_df = feat_df.slice(n_train)

    return tr_df, test_df