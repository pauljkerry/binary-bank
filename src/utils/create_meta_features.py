import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib


def create_meta_features(ID_list, level, fe_version=None, scale=False):
    """
    stackingのためのメタ特徴量を作成する

    Parameters
    ----------
    ID_list : list
        使用する予測値のID
    level : str
        predsの階層, "base", "l1" or "l2"
    fe_version : str, default None
        特徴量エンジニアリングのversion名
    scale : bool, default False
        予測値部分に標準化を適用するかどうか

    Returns
    -------
    tr_df : pd.DataFrame
        メタ特徴量の学習用データ
    test_df : pd.DataFrame
        メタ特徴量のテスト用データ

    Notes
    -----
    - scale=Trueにすると予測値部分にStandardScalerを適用
    """
    # 予測値の読み込みと結合
    tr_list = []
    test_list = []

    for ID in ID_list:
        array_tr = np.load(f"../artifacts/preds/{level}/oof_single_{ID}.npy")
        array_test = np.load(f"../artifacts/preds/{level}/test_single_{ID}.npy")

        if array_tr.ndim == 1:
            columns = [f"pred_{ID}"]
        else:
            n_classes = array_tr.shape[1]
            columns = [f"pred_{ID}_{i}" for i in range(n_classes)]

        tr_list.append(pd.DataFrame(array_tr, columns=columns))
        test_list.append(pd.DataFrame(array_test, columns=columns))

    tr_pred_df = pd.concat(tr_list, axis=1)
    test_pred_df = pd.concat(test_list, axis=1)

    # スケーリング（予測値のみ）
    if scale:
        scaler = StandardScaler()
        tr_pred_df = pd.DataFrame(
            scaler.fit_transform(tr_pred_df),
            columns=tr_pred_df.columns
        )
        test_pred_df = pd.DataFrame(
            scaler.transform(test_pred_df),
            columns=test_pred_df.columns
        )

    # 特徴量エンジニアリングの追加
    if fe_version is not None:
        tr_fe = pd.read_parquet(
            f"../artifacts/features/base/tr_df{fe_version}.parquet"
        )
        test_fe = pd.read_parquet(
            f"../artifacts/features/base/test_df{fe_version}.parquet"
        )
        tr_df = pd.concat([tr_fe, tr_pred_df], axis=1)
        test_df = pd.concat([test_fe, test_pred_df], axis=1)
    else:
        tr_df = tr_pred_df
        test_df = test_pred_df

    # targetの追加
    train_data = pd.read_parquet("../artifacts/prepro/train_data3.parquet")
    le_loaded = joblib.load("../artifacts/label_encoder.pkl")
    tr_df["target"] = le_loaded.transform(train_data["target"])

    return tr_df, test_df