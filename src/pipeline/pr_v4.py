import pandas as pd
from sklearn.preprocessing import PowerTransformer


def preprocessing():
    """
    生データの前処理を行い保存する関数。

    Notes
    -----
    - id列を削除
    - 特徴量名をリネーム
    - yeo-johnson変換
    - 外れ値を1%分位点でClip
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
        upper = all_data[col].quantile(0.99)
        lower = all_data[col].quantile(0.01)
        all_data[col] = all_data[col].clip(lower=lower, upper=upper)

        all_data[col] = pt.fit_transform(all_data[[col]])

    tr_df = all_data.iloc[:len(train_renamed)]
    test_df = all_data.iloc[len(train_renamed):]

    tr_df.to_parquet("../artifacts/prepro/train_data4.parquet")
    test_df.to_parquet("../artifacts/prepro/test_data4.parquet")

    print("data saved successfully!")