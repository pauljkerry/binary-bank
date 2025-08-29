import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def create_meta_features(
    oof_list: list[str],
    test_list: list[str],
    fe_version=None,
    scale=False
):
    """
    stackingのためのメタ特徴量を作成する

    Parameters
    ----------
    oof_list : list[np.ndarray]
        oofを格納したリスト
    test_list : list[np.ndarray]
        test predsを格納したリスト
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
    columns = [f"model_{i}" for i in range(len(oof_list))]
    oof_df = pd.DataFrame(np.array(oof_list).T, columns=columns)
    test_df = pd.DataFrame(np.array(test_list).T, columns=columns)

    # スケーリング（予測値のみ）
    if scale:
        scaler = StandardScaler()
        oof_df = pd.DataFrame(
            scaler.fit_transform(oof_df),
            columns=columns
        )
        test_df = pd.DataFrame(
            scaler.transform(test_df),
            columns=columns
        )

    # 特徴量エンジニアリングの追加
    if fe_version is not None:
        tr_fe = pd.read_parquet(
            f"../artifacts/features/base/tr_df{fe_version}.parquet"
        )
        test_fe = pd.read_parquet(
            f"../artifacts/features/base/test_df{fe_version}.parquet"
        )
        tr_df = pd.concat([tr_fe, oof_df], axis=1)
        test_df = pd.concat([test_fe, test_df], axis=1)
    else:
        tr_df = oof_df
        test_df = test_df

    # targetの追加
    y = np.load("../artifacts/y_true.npy")
    tr_df["target"] = y

    return tr_df, test_df