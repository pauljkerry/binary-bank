import polars as pl
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm.notebook import tqdm


def target_encoding(
    tr_df: pl.DataFrame,
    test_df: pl.DataFrame,
    target_col="target",
    cat_cols=None,
    n_splits=5,
    seed=42
):
    """
    Target Encodingを行う関数

    Parameters
    ----------
    tr_df : pl.DataFrame
        Training data
    test_df : pl.DataFrame
        Unlabeled data
    target_col : str
        Targetカラムの列名
    cat_cols : list
        カテゴリ変数の列名のリスト
    n_splits : int
        SKFの分割数
    seed : int
        Random seed

    Returns
    -------
    te_df : pl.DataFrame
        Target Encodingを行ったDF
    """
    """
    if isinstance(tr_df, pd.DataFrame):
        tr_df = pl.from_pandas(tr_df)
    elif isinstance(tr_df, pl.DataFrame):
        tr_df = tr_df
    else:
        raise TypeError("Expected pandas.DataFrame or polars.DataFrame")
    """

    tr_df = tr_df.with_row_count("id")
    test_df = test_df.with_row_count("id")

    y = tr_df[target_col].to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    if cat_cols is None:
        cat_cols = [c for c, t in tr_df.schema.items() if t == pl.Utf8 and c != target_col]

    te_tr_dict = {f"{col}_te": np.zeros(tr_df.height) for col in cat_cols}
    te_test_dict = {f"{col}_te": np.zeros(test_df.height) for col in cat_cols}

    for fold_idx, (tr_idx, val_idx) in enumerate(
        tqdm(skf.split(tr_df.to_pandas(), y))
    ):
        val_pl = tr_df[val_idx]
        train_pl = tr_df[tr_idx]

        for col in tqdm(cat_cols):
            # 1. trainデータでグループごとの平均計算（Polars）
            means_df = (
                train_pl.group_by(col)
                .agg(pl.col(target_col).mean())
                .rename({target_col: "mean_target"})
            )

            # 2. valデータにマッピング（Polarsのjoinで結合）
            val_with_mean = val_pl.join(means_df, on=col, how="left").sort("id")

            # 3. マッピングできなかったものは平均値で補完
            overall_mean = means_df["mean_target"].mean()
            val_te = val_with_mean["mean_target"].fill_null(overall_mean).to_numpy()

            te_tr_dict[f"{col}_te"][val_idx] = val_te

            # 4. テストデータも同様にjoin
            test_with_mean = test_df.join(means_df, on=col, how="left").sort("id")
            test_te = test_with_mean["mean_target"].fill_null(overall_mean).to_numpy()

            te_test_dict[f"{col}_te"] += test_te / n_splits

    te_tr = pl.DataFrame(te_tr_dict).with_columns([
        pl.col(col).cast(pl.Float32) for col in te_tr_dict.keys()
    ])
    te_test = pl.DataFrame(te_test_dict).with_columns([
        pl.col(col).cast(pl.Float32) for col in te_test_dict.keys()
    ])

    return pl.concat([te_tr, te_test], how="vertical")