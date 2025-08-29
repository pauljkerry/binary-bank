import pandas as pd


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
    sub_df = pd.read_csv("../input/sample_submission.csv")
    sub_df.iloc[:, 1] = preds

    sub_df.to_csv(f"../output/submission_{ID}.csv", index=False)
    print("Saved submission file successfully!")