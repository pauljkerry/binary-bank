import polars as pl
import pandas as pd
from src.utils.target_encoding import target_encoding
from itertools import combinations
from sklearn.preprocessing import StandardScaler


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
    - GBDT用
    - 2変数TE
    """
    # === 初期情報 ===
    train_len = len(train_data)

    test_data = test_data.with_columns([
        pl.lit(0).cast(pl.Int64).alias("target")
    ])

    all_data = pl.concat([train_data, test_data])

    # === 1) 数値特徴量（そのまま） ===
    numeric_dtypes = {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64}
    num_cols = [col for col, dtype in zip(all_data.columns, all_data.dtypes) if dtype in numeric_dtypes]
    num_df = all_data.select(num_cols)

    # === 2) 単体のカテゴリ特徴量のTarget Encoding ===
    te_single = target_encoding(train_data, test_data)

    # === 3) 全列を文字列化して、2変数の交互作用を作成 ===
    str_all_data = all_data.select([pl.col(c).cast(pl.Utf8) for c in all_data.columns])
    inter_exprs2 = [
        pl.format("{}_{}", pl.col(col1), pl.col(col2)).alias(f"{col1}_{col2}")
        for col1, col2 in combinations(str_all_data.columns, 2)
        if "target" not in (col1, col2)   # targetは除外
    ]
    inter_df2 = str_all_data.select(inter_exprs2)
    inter_train = inter_df2[:train_len]
    inter_train = inter_train.with_columns(
        train_data["target"].alias("target")
    )
    inter_test = inter_df2[train_len:]
    te_inter2 = target_encoding(inter_train, inter_test)

    # === 4) 全特徴量を結合 ===
    df_feat = pl.concat([num_df, te_single, te_inter2], how="horizontal")

    # ==== 5) 標準化 ===
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df_feat.to_pandas())
    scaled_df = pd.DataFrame(
        scaled_array.astype("float32"),
        columns=df_feat.columns
    )
    scaled_df = pl.from_pandas(scaled_df)

    # === 5) 再分割 ===
    tr_df = scaled_df[:train_len]
    test_df = scaled_df[train_len:]

    # === 6) target列を戻す ===
    tr_df = tr_df.with_columns(train_data["target"])

    return tr_df, test_df