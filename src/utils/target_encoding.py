from typing import Optional
import polars as pl
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm.notebook import tqdm


def target_encoding(
    tr_df: pl.DataFrame,
    test_df: pl.DataFrame,
    target: str = "target",
    cat_cols: Optional[list[str]] = None,
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

    if cat_cols is None:
        cat_cols = [
            c
            for c, t in tr_df.schema.items()
            if t == pl.Utf8 and c != target
        ]

    te_tr_dict = {
        f"{col}_te": np.zeros(tr_df.height, dtype=np.float32)
        for col in cat_cols
    }
    te_test_dict = {
        f"{col}_te": np.zeros(test_df.height, dtype=np.float32)
        for col in cat_cols
    }

    for fold_idx, (tr_idx, val_idx) in enumerate(
        tqdm(skf.split(np.zeros_like(y), y))
    ):
        train = tr_df[tr_idx]
        val = tr_df[val_idx]

        for col in tqdm(cat_cols):
            means_df = (
                train.select([col, target])
                .group_by(col)
                .agg(pl.col(target).mean().alias("mean_target"))
            )

            # validation
            overall_mean = means_df["mean_target"].mean()
            val_te = (
                val.join(
                    means_df.select(["mean_target", col]),
                    on=col,
                    how="left"
                )
                .get_column("mean_target")
                .fill_null(overall_mean)
                .to_numpy()
                .astype(dtype=np.float32, copy=False)
            )
            te_tr_dict[f"{col}_te"][val_idx] = val_te

            # test
            test_te = (
                test_df.join(
                    means_df.select(["mean_target", col]),
                    on=col,
                    how="left"
                )
                .get_column("mean_target")
                .fill_null(overall_mean)
                .to_numpy()
                .astype(dtype=np.float32, copy=False)
            )
            te_test_dict[f"{col}_te"] += test_te / n_splits

            del means_df, val_te, test_te
        del train, val

    te_tr = pl.DataFrame(te_tr_dict)
    te_test = pl.DataFrame(te_test_dict)

    return pl.concat([te_tr, te_test], how="vertical")