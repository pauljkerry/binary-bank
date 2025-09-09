import re, glob
from pathlib import Path
from typing import List, Tuple, Dict, Optional

_FOLD_RE = re.compile(r"-fold(\d+)-seed(\d+)\.parquet$")


def make_fold_paths(
    base_dir: str | Path,
    ID: str | int,
    SEED: int,
    valid_fold_idx: int = 0,
    n_splits: Optional[int] = None,
) -> Tuple[List[str], str, Dict]:
    """
    指定ID/SEEDの Parquet を列挙し、train/valid に仕分ける。
    Returns:
      train_files, valid_file, info(dict: {'train_idx': [...], 'valid_idx': [...], 'all_files': [...]})
    Raises:
      FileNotFoundError, ValueError
    """
    base_dir = Path(base_dir)
    pattern = str(base_dir / f"tr_df{ID}-fold*-seed{SEED}.parquet")
    all_files = glob.glob(pattern)
    if not all_files:
        raise FileNotFoundError(f"No files matched: {pattern}")

    def fold_key(p: str) -> int:
        m = _FOLD_RE.search(Path(p).name)
        return int(m.group(1)) if m else 10**9

    all_files.sort(key=fold_key)

    train_files: List[str] = []
    valid_file: Optional[str] = None
    train_idx, valid_idx = [], []

    for p in all_files:
        m = _FOLD_RE.search(Path(p).name)
        if not m:
            continue
        f = int(m.group(1))
        seed_in_name = int(m.group(2))
        if seed_in_name != SEED:
            continue
        if f == valid_fold_idx:
            valid_file = p
            valid_idx.append(f)
        else:
            train_files.append(p)
            train_idx.append(f)

    if valid_file is None:
        raise FileNotFoundError(f"Valid fold file not found for fold={valid_fold_idx} under {pattern}")

    if n_splits is not None:
        expected = set(range(n_splits))
        present = set(train_idx + valid_idx)
        missing = expected - present
        if missing:
            raise FileNotFoundError(f"Missing folds: {sorted(missing)} (present: {sorted(present)})")

    info = {
        "train_idx": train_idx,
        "valid_idx": valid_idx,
        "all_files": all_files,
        "pattern": pattern,
    }
    return train_files, valid_file, info

"""
W&B 連携の小ネタ

wandb.config.update(info) で pattern, train_idx, valid_idx, all_files を記録

生成した train_files.txt, valid_file.txt を wandb.Artifact として保存しておくと、後で完全再現が楽
"""