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
    original_data = pd.read_parquet("../input/original.parquet")

    train_data = train_data.copy()
    test_data = test_data.copy()
    original_data = original_data.copy()

    rename_map = {
        "y": "target"
    }

    train_renamed = train_data.rename(
        columns=rename_map).drop("id", axis=1, errors="ignore")
    test_renamed = test_data.rename(
        columns=rename_map).drop("id", axis=1, errors="ignore")
    original_renamed = original_data.rename(
        columns=rename_map).drop("id", axis=1, errors="ignore")

    original_renamed["target"] = original_renamed["target"].map({"yes": 1, "no": 0}).astype("int64")

    tr_df = pd.concat([train_renamed, original_renamed], axis=0, ignore_index=True)
    tr_df.to_parquet("../artifacts/prepro/train_data2.parquet")
    test_renamed.to_parquet("../artifacts/prepro/test_data2.parquet")

    print("data saved successfully!")