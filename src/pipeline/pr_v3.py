import pandas as pd
from sklearn.preprocessing import PowerTransformer


def preprocessing():
    """
    生データの前処理を行い保存する関数。

    Notes
    -----
    - id列を削除
    - 特徴量名をリネーム
    - 数値変数の分布の歪みを補正
    """
    train_data = pd.read_csv("../input/train.csv")
    test_data = pd.read_csv("../input/test.csv")

    train_data = train_data.copy()
    test_data = test_data.copy()

    rename_map = {
        "y": "target"
    }

    train_renamed = train_data.rename(
        columns=rename_map).drop("id", axis=1, errors="ignore")
    test_renamed = test_data.rename(
        columns=rename_map).drop("id", axis=1, errors="ignore")

    all_data = pd.concat([train_renamed, test_renamed], axis=0)

    all_data["previous"] = (all_data["previous"] > 0).astype(int)
    all_data["pdays"] = (all_data["pdays"] != -1).astype(int)

    pt = PowerTransformer(method="yeo-johnson")

    for col in ["balance", "campaign", "duration"]:
        all_data[col] = pt.fit_transform(all_data[[col]])

    tr_df = all_data.iloc[:len(train_renamed)]
    test_df = all_data.iloc[len(train_renamed):]

    tr_df.to_parquet("../artifacts/prepro/train_data3.parquet")
    test_df.to_parquet("../artifacts/prepro/test_data3.parquet")

    print("data saved successfully!")