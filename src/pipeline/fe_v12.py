import polars as pl
from itertools import combinations
from tqdm.notebook import tqdm
import numpy as np


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
    tr_df12 = pl.read_parquet("../artifacts/features/base/tr_df12.parquet").drop("target")
    test_df12 = pl.read_parquet("../artifacts/features/base/test_df12.parquet")
    all_data12 = pl.concat([tr_df12, test_df12], how="vertical")

    train_len = 750000
    orig_data = train_data[train_len:]
    train_data2 = train_data[:train_len]

    test_data = test_data.with_columns([
        pl.lit(0).cast(pl.Int64).alias("target")
    ])

    all_data = pl.concat([train_data2, test_data])

    cat_cols = [col for col, dtype in zip(all_data.columns, all_data.dtypes) if dtype == pl.Utf8]
    # === 3) 全列を文字列化して、2変数の交互作用を作成 ===
    str_all_data = all_data.select([pl.col(c).cast(pl.Utf8) for c in all_data.columns])
    inter_exprs2 = [
        pl.format("{}_{}", pl.col(col1), pl.col(col2)).alias(f"{col1}_{col2}")
        for col1, col2 in combinations(str_all_data.columns, 2)
        if "target" not in (col1, col2)   # targetは除外
    ]
    inter_df2 = str_all_data.select(inter_exprs2)
    inter_df1 = str_all_data.select(cat_cols)

    inter_all = pl.concat([inter_df1, inter_df2], how="horizontal")

    # origのtargetを使ってTE
    str_orig_data = orig_data.select([pl.col(c).cast(pl.Utf8) for c in orig_data.columns])
    inter_exprs3 = [
        pl.format("{}_{}", pl.col(col1), pl.col(col2)).alias(f"{col1}_{col2}")
        for col1, col2 in combinations(str_orig_data.columns, 2)
        if "target" not in (col1, col2)   # targetは除外
    ]
    inter_df3 = str_orig_data.select(inter_exprs3)
    inter_df3 = inter_df3[:train_len].with_columns(
        orig_data["target"].alias("target")
    )
    inter_df4 = str_all_data.select(cat_cols)
    inter_all2 = pl.concat([inter_df3, inter_df4], how="horizontal")

    cat_cols = [c for c, t in inter_all2.schema.items() if t == pl.Utf8 and c != "target"]

    te_dict = {f"{col}_te2": np.zeros(inter_all.height) for col in cat_cols}

    for col in tqdm(cat_cols):
        # 1. trainデータでグループごとの平均計算（Polars）
        means_df = (
            inter_all2.group_by(col)
            .agg(pl.col("target").mean())
            .rename({"target": "mean_target"})
        )

        # 2. valデータにマッピング（Polarsのjoinで結合）
        all_data_mean = inter_all.join(means_df, on=col, how="left")

        # 3. マッピングできなかったものは平均値で補完
        overall_mean = means_df["mean_target"].mean()
        all_te = all_data_mean["mean_target"].fill_null(overall_mean).to_numpy()

        te_dict[f"{col}_te2"] = all_te

    te_all = pl.DataFrame(te_dict).with_columns([
        pl.col(col).cast(pl.Float32) for col in te_dict.keys()
    ])

    # === 4) 全特徴量を結合 ===
    df_feat = pl.concat([all_data12, te_all], how="horizontal")

    # === 5) 再分割 ===
    tr_df = df_feat[:train_len]
    test_df = df_feat[train_len:]

    # === 6) target列を戻す ===
    tr_df = tr_df.with_columns(train_data2["target"])

    return tr_df, test_df