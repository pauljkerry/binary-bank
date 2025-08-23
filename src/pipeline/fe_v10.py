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
    - 3変数TEのみ
    - Chunk処理
    """
    # === 初期情報 ===
    chank_to_process = 3  # 0~3まで
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
    train_data = all_data[:train_len]
    test_data = all_data[train_len:]

    # === 3変数の交互作用 ===
    cols = [c for c in all_data.columns if c != "target"]
    all_combs = list(combinations(cols, 3))
    chunk_size = len(all_combs) // 4 + 1
    comb_chunks = [all_combs[i:i+chunk_size] for i in range(0, len(all_combs), chunk_size)]

    # 2. チャンクごとに処理
    for i, chunk in enumerate(comb_chunks):
        if i != chank_to_process:
            continue
        # 交互作用列を作る
        inter_exprs = [
            pl.format("{}_{}_{}", pl.col(c1), pl.col(c2), pl.col(c3)).alias(f"{c1}_{c2}_{c3}")
            for c1, c2, c3 in chunk
        ]
        inter_df_chunk = all_data.select(inter_exprs)

        # train/test に分割
        inter_train = inter_df_chunk[:train_len].with_columns(train_data["target"].alias("target"))
        inter_test = inter_df_chunk[train_len:]

        # このチャンクだけTE
        te_chunk = target_encoding(inter_train, inter_test)

    print("phase6")

    # === 5) 再分割 ===
    tr_df = te_chunk[:train_len]
    test_df = te_chunk[train_len:]

    # === 6) target列を戻す ===
    tr_df = tr_df.with_columns(train_data["target"])

    return tr_df, test_df