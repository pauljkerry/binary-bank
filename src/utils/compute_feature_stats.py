import numpy as np
import polars as pl


def compute_feature_stats(
    paths: list[str],
    features: list[str],
    num_cols: list[str]
) -> (np.ndarray, np.ndarray):
    lf = pl.scan_parquet(paths, low_memory=True)

    exprs = []
    for c in num_cols:
        exprs += [pl.col(c).cast(pl.Float32).mean().alias(f"{c}_mean"),
                  pl.col(c).cast(pl.Float32).std(ddof=0).alias(f"{c}_std")]
    out = lf.select(exprs).collect(streaming=True)
    mean = out.select(
        [f"{c}_mean" for c in num_cols]).to_numpy().ravel().astype(np.float32)
    std = out.select(
        [f"{c}_std" for c in num_cols]).to_numpy().ravel().astype(np.float32)
    std[std == 0] = 1.0
    return mean, std