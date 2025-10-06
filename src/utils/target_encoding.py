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
    alpha: int = 0,
    n_splits: int = 5,
    seed: int = 42
) -> pl.DataFrame:
    """
    Out-of-fold (OOF) target encoding with M-estimate smoothing.

    For each column in `key_cols`, per-category statistics are computed on the
    fold's training split and joined to the validation split (leak-free).
    Test features are computed per fold and averaged. For "mean", the smoothed
    estimate is:
        (n * mean + alpha * global_mean) / (n + alpha)
    Unseen categories are filled with the fold's global statistics.

    Parameters
    ----------
    tr_df : pl.DataFrame
        Training data. Must contain `target` and all `key_cols`.
    test_df : pl.DataFrame
        Unlabeled data. Must contain all `key_cols`.
    key_cols : list[str]
        Discrete/categorical keys used for grouping. Do not pass raw float
        columns; round/bin or stringify them first to avoid join drift.
    target : str, default "target"
        Target column name. For "count", this function counts positives as
        `(target == 1).sum()` (binary assumption).
    stats : tuple of {"mean","std","min","max","median","count"}, default (...)
        Per-category statistics to output. Only "mean" is smoothed by `alpha`.
        ("count" means positive count for binary targets.)
    alpha : float, default 20.0
        Smoothing strength (half-life). `alpha=0` disables smoothing.
        n≈alpha ⇒ the category mean is trusted ~50%.
    n_splits : int, default 5
        Number of StratifiedKFold splits.
    seed : int, default 42
        Random seed for fold shuffling.

    Returns
    -------
    pl.DataFrame
        Encoded features for train and test stacked vertically. Columns are
        named `{target}_{stat}_by_{col}` in the order of `key_cols`.
        Shape: (tr_df.height + test_df.height, sum_over_cols len(stats_for_col))

    Notes
    -----
    - OOF computation prevents target leakage.
    - Unseen categories are filled with fold-wise global stats (e.g., global_mean).
    - If you need raw frequency n_i, add an explicit aggregation; "count" here
      is the positive-class count (binary).
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
        global_mean = train.select(pl.col(target).mean()).to_series()[0]

        base = train.select([
            pl.col(target).mean().alias("mean"),
            pl.col(target).std(ddof=1).alias("std"),
            pl.col(target).min().alias("min"),
            pl.col(target).max().alias("max"),
            pl.col(target).median().alias("median"),
            (pl.col(target) == 1).sum().alias("cnt"),
        ]).to_dicts()[0]

        for col in tqdm(key_cols):
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
                    (
                        (pl.col(target).sum() + pl.lit(alpha) * pl.lit(global_mean))
                        / (pl.len() + pl.lit(alpha))
                    ).alias(f"{target}_mean_by_{col}")
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