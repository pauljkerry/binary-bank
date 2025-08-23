import polars as pl
from src.utils.target_encoding import target_encoding
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
    - GBDT用
    - 1変数 + 2変数TE
    """
    # === 初期情報 ===
    train_len = len(train_data)
    print("phase1")

    # === test dataにtargetを追加 ===
    test_data = test_data.with_columns([
        pl.lit(0).cast(pl.Int64).alias("target")
    ])
    print("phase2")

    # === Dataの結合とTarget以外をカテゴリ変数に ===
    all_data = pl.concat([train_data, test_data])
    all_data = all_data.select([
        pl.col(c).cast(pl.Utf8) if c != "target" else pl.col(c)
        for c in all_data.columns
    ])
    print("phase3")

    # === 1変数の Target Encoding ===
    train_data = all_data[:train_len]
    test_data = all_data[train_len:]
    te_single = target_encoding(train_data, test_data)
    print("phase4")

    # === 2変数の交互作用 ===
    inter_exprs2 = [
        pl.format("{}_{}", pl.col(col1), pl.col(col2)).alias(f"{col1}_{col2}")
        for col1, col2 in combinations(all_data.columns, 2)
        if "target" not in (col1, col2)   # targetは除外
    ]
    inter_df2 = all_data.select(inter_exprs2)
    inter_train = inter_df2[:train_len]
    inter_train = inter_train.with_columns(
        train_data["target"].alias("target")
    )
    inter_test = inter_df2[train_len:]
    te_inter2 = target_encoding(inter_train, inter_test)
    print("phase5")

    # === すべて結合 ===
    df_feat = pl.concat([te_single, te_inter2], how="horizontal")
    print("phase6")

    # === 5) 再分割 ===
    tr_df = df_feat[:train_len]
    test_df = df_feat[train_len:]

    # === 6) target列を戻す ===
    tr_df = tr_df.with_columns(train_data["target"])

    return tr_df, test_df