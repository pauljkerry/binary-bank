import os
import numpy as np
import pyarrow.dataset as ds

def collect_row_ids_for_fold(paths, fold_col, fold_idx, batch_rows=1_000_000):
    if isinstance(paths, (str, os.PathLike)):
        paths = [str(paths)]
    dataset = ds.dataset(paths, format="parquet")
    reader = dataset.scanner(
        columns=["row_id"],
        filter=(ds.field(fold_col) == fold_idx),
        batch_size=batch_rows,
    ).to_reader()

    chunks = []
    for batch in reader:
        # Arrow Array -> numpy
        arr = batch.column(0)  # row_id
        chunks.append(np.asarray(arr))
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)