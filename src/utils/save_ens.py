import numpy as np


def save_ensemble_prediction(test_preds_list, weights_dict, ID):
    """
    アンサンブル予測を計算して保存する。

    Parameters
    ----------
    test_preds_list : list of np.ndarray
        各モデルのテストデータ予測値のリスト
    weights_dict : dict
        比率のパラメータ
    ID : str
        保存ファイルに使用する識別子
    """
    # 重みを順に取り出して正規化
    weights = np.array([weights_dict[f"raw_w{i}"]
                        for i in range(len(test_preds_list))])
    weights /= weights.sum()

    prob_list = []
    for test_preds in test_preds_list:
        if test_preds.ndim == 1:
            # すでにクラス1の予測（MLPなど）
            prob = test_preds
        elif test_preds.shape[1] == 1:
            # 1列2次元（安全対策）
            prob = test_preds[:, 0]
        else:
            # predict_proba 出力（CatBoostなど）
            prob = test_preds[:, 1]  # クラス1の確率
        prob_list.append(prob)

    # 加重平均
    y_prob = sum(w * p for w, p in zip(weights, prob_list))

    # 確率→クラス変換
    y_pred = (y_prob > 0.5).astype(int)

    save_path = f"../artifacts/preds/ens/ens_{ID}.npy"

    np.save(save_path, y_pred)
    print("Ensemble preds saved successfully")