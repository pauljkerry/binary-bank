from typing import Optional
import polars as pl
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm.notebook import tqdm


def target_encoding(
    tr_df: pl.DataFrame,
    test_df: pl.DataFrame,
    key_cols: list[str],
    target: str = "target",
    stats: tuple[str, ...] = ("mean", "std", "min", "max", "median", "count"),
    n_splits: int = 5,
    seed: int = 42
):
    """
    Target Encodingを行う関数

    Parameters
    ----------
    tr_df : pl.DataFrame
        Training data
    test_df : pl.DataFrame
        Unlabeled data
    target : str
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
    y = tr_df.get_column(target).to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def stat_names(col: str) -> list[str]:
        names = []
        if "mean" in stats:
            names.append(f"{target}_mean_by_{col}")
        if "std" in stats:
            names.append(f"{target}_std_by_{col}")
        if "min" in stats:
            names.append(f"{target}_min_by_{col}")
        if "max" in stats:
            names.append(f"{target}_max_by_{col}")
        if "median" in stats:
            names.append(f"{target}_median_by_{col}")
        if "count" in stats:
            names.append(f"{target}_count_by_{col}")  # 1の個数
        return names

    all_cols = []
    for col in key_cols:
        all_cols.extend(stat_names(col))

    N_tr, N_te = tr_df.height, test_df.height

    te_train = {c: np.zeros(N_tr, dtype=np.float32) for c in all_cols}
    te_test = {c: np.zeros(N_te, dtype=np.float32) for c in all_cols}

    for fold_idx, (tr_idx, val_idx) in enumerate(
        tqdm(skf.split(np.zeros_like(y), y))
    ):
        train = tr_df[tr_idx]
        val = tr_df[val_idx]

        for col in tqdm(key_cols):
            base = train.select([
                pl.col(target).mean().alias("mean"),
                pl.col(target).std(ddof=1).alias("std"),
                pl.col(target).min().alias("min"),
                pl.col(target).max().alias("max"),
                pl.col(target).median().alias("median"),
                (pl.col(target) == 1).sum().alias("cnt"),
            ]).to_dicts()[0]

            fill_map = {}
            for s in stats:
                name = f"{target}_{s}_by_{col}"
                if s == "count":
                    fill_map[name] = 0.0
                else:
                    fill_map[name] = float(base[s])

            aggs = []
            col_names = []
            if "mean" in stats:
                aggs.append(
                    pl.col(target).mean().alias(f"{target}_mean_by_{col}")
                )
                col_names.append(f"{target}_mean_by_{col}")
            if "std" in stats:
                aggs.append(
                    pl.col(target).std(ddof=1).alias(f"{target}_std_by_{col}")
                )
                col_names.append(f"{target}_std_by_{col}")
            if "min" in stats:
                aggs.append(
                    pl.col(target).min().alias(f"{target}_min_by_{col}")
                )
                col_names.append(f"{target}_min_by_{col}")
            if "max" in stats:
                aggs.append(
                    pl.col(target).max().alias(f"{target}_max_by_{col}")
                )
                col_names.append(f"{target}_max_by_{col}")
            if "median" in stats:
                aggs.append(
                    pl.col(target).median().alias(f"{target}_median_by_{col}")
                )
                col_names.append(f"{target}_median_by_{col}")
            if "count" in stats:
                aggs.append(
                    (pl.col(target) == 1).sum().alias(f"{target}_count_by_{col}")
                )
                col_names.append(f"{target}_count_by_{col}")

            grouped_df = (
                train.select([col, target])
                .group_by(col)
                .agg(aggs)
            )

            # validation
            val_mat = (
                val.join(
                    grouped_df.select(col_names + [col]),
                    on=col,
                    how="left"
                )
                .select(col_names)
                .with_columns(
                    [
                        pl.col(c).fill_null(fill_map[c]).alias(c)
                        for c in col_names
                    ]
                )
                .to_numpy()
                .astype(dtype=np.float32, copy=False)
            )

            for j, name in enumerate(col_names):
                te_train[name][val_idx] = val_mat[:, j]

            # test
            test_mat = (
                test_df.join(
                    grouped_df.select(col_names + [col]),
                    on=col,
                    how="left"
                )
                .select(col_names)
                .with_columns(
                    [
                        pl.col(c).fill_null(fill_map[c]).alias(c)
                        for c in col_names
                    ]
                )
                .to_numpy()
                .astype(dtype=np.float32, copy=False)
            )
            for j, name in enumerate(col_names):
                te_test[name] += test_mat[:, j] / n_splits

            del grouped_df, val_mat, test_mat
        del train, val

    te_tr = pl.DataFrame(te_train)
    te_test = pl.DataFrame(te_test)

    return pl.concat([te_tr, te_test], how="vertical")