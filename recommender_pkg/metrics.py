import numpy as np


def dcg_score(l):
    """TODO: write docstring"""
    return sum([s/np.log2(i+2) for i, s in enumerate(l)])


def ndcg_score(l):
    """TODO: write docstring"""
    idcg = dcg_score(sorted(l, reverse = True))
    return idcg if idcg == 0 else dcg_score(l) / idcg


def perform_groupwise_evaluation(X_test, y_test, y_pred):
    """TODO: write docstring"""

    # TODO: add kwdarg for groups
    # TODO: add kwdarg for metrics to return

    hits = np.empty(0)
    gains = np.empty(0)

    for user in np.unique(X_test[:, 0]):
        user_idx = X_test[:, 0] == user
        X_test_temp = X_test[user_idx]
        y_test_temp = y_test[user_idx]
        y_pred_temp = y_pred[user_idx]
        idx_top = np.argsort(y_pred_temp)[:-11:-1]
        interactions_top = y_test_temp[idx_top]
        hits = np.append(hits, interactions_top.sum())
        gains = np.append(gains, ndcg_score(interactions_top))

    return {"hit_ratio": hits.mean(),
            "normalized_discounted_cumulative_gain": gains.mean()}
