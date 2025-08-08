import polars as pl
import pandas as pd
import numpy as np
from src.utils.target_encoding import target_encoding
from itertools import combinations


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
    # === 初期情報 ===
    train_len = len(train_data)

    # === polarsに変換 ===
    pl_train = pl.from_pandas(train_data)
    pl_test = pl.from_pandas(test_data)

    pl_test = pl_test.with_columns([
        pl.lit(0).cast(pl.Int64).alias("target")
    ])


    # === targetを除いて結合 ===
    all_data = pl.concat([pl_train, pl_test])

    # === 1) 数値特徴量（そのまま） ===
    numeric_dtypes = {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64}
    num_cols = [col for col, dtype in zip(all_data.columns, all_data.dtypes) if dtype in numeric_dtypes]
    num_df = all_data.select(num_cols).to_pandas().reset_index(drop=True)

    # === 2) 単体のカテゴリ特徴量の Target Encoding ===
    te_single = target_encoding(train_data, test_data).reset_index(drop=True)

    # === 3) 全列を文字列化して、2変数の交互作用を作成 ===
    str_all_data = all_data.select([pl.col(c).cast(pl.Utf8) for c in all_data.columns])
    inter_exprs = []
    for col1, col2 in combinations(str_all_data.columns, 2):
        inter_exprs.append((pl.col(col1) + "_" + pl.col(col2)).alias(f"{col1}_x_{col2}"))

    inter_df = str_all_data.select(inter_exprs).to_pandas()
    inter_train = inter_df.iloc[:train_len].copy()
    inter_train["target"] = train_data["target"]
    inter_test = inter_df.iloc[train_len:].copy()
    te_inter = target_encoding(inter_train, inter_test).reset_index(drop=True)

    # === 4) 全特徴量を結合 ===
    df_feat = pd.concat([num_df, te_single, te_inter], axis=1)

    # === 5) 再分割 ===
    tr_df = df_feat.iloc[:train_len].copy()
    test_df = df_feat.iloc[train_len:].copy()

    # === 6) target列を戻す ===
    tr_df["target"] = train_data["target"]

    return tr_df, test_df