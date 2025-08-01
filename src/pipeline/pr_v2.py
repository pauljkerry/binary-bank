import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def preprocessing():
    """
    生データの前処理を行い保存する関数。

    Notes
    -----
    - id列を削除
    - 特徴量名をリネーム
    - train dataとoriginal dataを結合
    - 欠損値を中央値と最頻値で補完
    """
    train_data = pd.read_csv("../input/train.csv")
    test_data = pd.read_csv("../input/test.csv")

    train_data = train_data.copy()
    test_data = test_data.copy()

    rename_map = {
        "ABC": "target"
    }

    train_renamed = train_data.rename(
        columns=rename_map).drop("id", axis=1, errors="ignore")
    test_renamed = test_data.rename(
        columns=rename_map).drop("id", axis=1, errors="ignore")

    all_data = pd.concat([
        train_renamed,
        test_renamed], axis=0)

    num_cols = all_data.select_dtypes(include=np.number).columns.difference(["target"])
    cat_cols = all_data.select_dtypes(include=["object", "category"]).columns.difference(["target"])

    # 数値列は中央値で補完
    num_imputer = SimpleImputer(strategy="median")
    all_data[num_cols] = num_imputer.fit_transform(all_data[num_cols])

    # カテゴリ列は"Unknown"で補完
    cat_imputer = SimpleImputer(strategy="constant", fill_value="Unknown")
    all_data[cat_cols] = cat_imputer.fit_transform(all_data[cat_cols])

    tr_df = all_data[:len(train_data)]
    test_df = all_data[len(train_data):]

    tr_df.to_parquet("../artifacts/prepro/train_data2.parquet")
    test_df.to_parquet("../artifacts/prepro/test_data2.parquet")

    print("data saved successfully!")