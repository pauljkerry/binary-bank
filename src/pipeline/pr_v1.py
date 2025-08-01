import pandas as pd


def preprocessing():
    """
    生データの前処理を行い保存する関数。

    Notes
    -----
    - id列を削除
    - 特徴量名をリネーム
    - train dataとoriginal dataを結合
    """
    train_data = pd.read_csv("../input/train.csv")
    test_data = pd.read_csv("../input/test.csv")

    train_data = train_data.copy()
    test_data = test_data.copy()

    rename_map = {
        "ABCDE": "target"
    }

    train_renamed = train_data.rename(
        columns=rename_map).drop("id", axis=1, errors="ignore")
    test_renamed = test_data.rename(
        columns=rename_map).drop("id", axis=1, errors="ignore")

    train_renamed.to_parquet("../artifacts/prepro/train_data1.parquet")
    test_renamed.to_parquet("../artifacts/prepro/test_data1.parquet")

    print("data saved successfully!")