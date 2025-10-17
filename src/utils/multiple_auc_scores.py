import cupy as cp


def multiple_auc_scores(
    y_true: cp.ndarray,
    y_scores: cp.ndarray
) -> float:
    """
    Parameters
    ----------
    y_true : cp.ndarray
        shape (n_samples,) 0/1のラベル
    y_score : cp.ndarray
        shape (n_samples,) 予測スコア（確率や連続値）

    Return
    ------
    auc : float
        AUCスコア
    """
    # スコアを昇順に並べ替えて順位をつける
    n_models = y_scores.shape[1]
    aucs = cp.zeros(n_models)
    for i in range(n_models):
        order = cp.argsort(y_scores[:, i])
        ranks = cp.empty_like(order, dtype=cp.float64)
        ranks[order] = cp.arange(1, len(y_scores[:, i])+1)

        # 正例のランクの合計
        pos_ranks = cp.sum(ranks[y_true == 1])
        n_pos = cp.sum(y_true == 1)
        n_neg = cp.sum(y_true == 0)

        # Mann-Whitney U 統計量から AUCを計算
        auc = (pos_ranks - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
        aucs[i] = auc
    return aucs
