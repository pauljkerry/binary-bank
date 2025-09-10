import re, glob
from pathlib import Path
from typing import List

_FOLD_RE = re.compile(r"-fold(\d+)-seed(\d+)\.parquet$")


def make_fold_paths(
    base_dir: str | Path,
    ID: str | int,
    SEED: int,
) -> [List[str]]:
    """
    指定ID/SEEDの Parquet を列挙し、train/valid に仕分ける。
    Returns:
      train_files
    Raises:
      FileNotFoundError, ValueError
    """
    base_dir = Path(base_dir)
    pattern = str(base_dir / f"tr_df{ID}-seed{SEED}.parquet")
    all_files = glob.glob(pattern)
    if not all_files:
        raise FileNotFoundError(f"No files matched: {pattern}")

    train_files: List[str] = []

    for p in all_files:
        m = _FOLD_RE.search(Path(p).name)
        if not m:
            continue
        f = int(m.group(1))
        seed_in_name = int(m.group(2))
        if seed_in_name != SEED:
            continue
        else:
            train_files.append(p)

    return train_files

"""
W&B 連携の小ネタ

wandb.config.update(info) で pattern, train_idx, valid_idx, all_files を記録

生成した train_files.txt, valid_file.txt を wandb.Artifact として保存しておくと、後で完全再現が楽
"""