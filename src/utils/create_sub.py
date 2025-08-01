import numpy as np
import pandas as pd
import joblib


def create_sub(preds, ID):
    """
    Kaggleの提出用のフォーマットに整形する関数。

    Parameters
    ----------
    test_proba : np.ndarray
        各ラベルについての予測値の配列。
    ID : str
        ファイルの識別子
    """
    le_loaded = joblib.load("../artifacts/label_encoder.pkl")

    label = le_loaded.inverse_transform(preds)

    sub_df = pd.DataFrame({
        "id": np.arange(18524, 18524 + len(preds)),
        "Personality": label
    })
    sub_df.to_csv(f"../output/submission_{ID}.csv", index=False)
    print("Saved submission file successfully!")