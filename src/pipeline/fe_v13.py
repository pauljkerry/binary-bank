import polars as pl
from itertools import combinations
from tqdm.notebook import tqdm
import numpy as np
import gc


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
    tr_df22 = pl.read_parquet("../artifacts/features/base/tr_df22.parquet").drop("target")
    test_df22 = pl.read_parquet("../artifacts/features/base/test_df22.parquet")
    all_data22 = pl.concat([tr_df22, test_df22], how="vertical")

    test_data = test_data.with_columns([
        pl.lit(0).cast(pl.Int64).alias("target")
    ])

    numeric_dtypes = {
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64
    }

    all_data = pl.concat([train_data, test_data])
    num_cols = [
        col for col, dtype in zip(all_data.columns, all_data.dtypes)
        if dtype in numeric_dtypes and col != "target"
    ]
    cat_cols = [col for col, dtype in zip(all_data.columns, all_data.dtypes) if dtype == pl.Utf8]

    # === 3) 全列を文字列化して、2変数の交互作用を作成 ===
    str_all_data = all_data.select([pl.col(c).cast(pl.Utf8) for c in all_data.columns])
    inter_df1 = str_all_data.select(cat_cols)
    inter_exprs2 = [
        pl.format("{}_{}", pl.col(col1), pl.col(col2)).alias(f"{col1}_{col2}")
        for col1, col2 in combinations(str_all_data.columns, 2)
        if "target" not in (col1, col2)   # targetは除外
    ]
    inter_df2 = str_all_data.select(inter_exprs2)

    inter_all = pl.concat([inter_df1, inter_df2], how="horizontal")

    cat_cols = [c for c in inter_all.columns if c != "target"]
    ce_dict = {f"{col}_ce": np.zeros(inter_all.height) for col in cat_cols}
    for col in tqdm(cat_cols):
        counts = inter_all.group_by(col).agg(pl.count().alias(f"{col}_ce"))
        joined_df = inter_all.join(counts, on=col, how="left")
        ce_dict[f"{col}_ce"] = joined_df[f"{col}_ce"]
        del joined_df
        gc.collect()

    ce_df = pl.DataFrame(ce_dict).with_columns([
        pl.col(col).cast(pl.Float32) for col in ce_dict.keys()
    ])

    # === 4) 全特徴量を結合 ===
    df_feat = pl.concat([all_data22, ce_df], how="horizontal")

    # === 5) 再分割 ===
    tr_df = df_feat[:len(train_data)]
    test_df = df_feat[len(train_data):]

    # === 6) target列を戻す ===
    tr_df = tr_df.with_columns(train_data["target"])

    return tr_df, test_df