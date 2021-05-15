import numpy as np


def dcg_score(items):
    """Calculate the discounted cumulative gain.

    Args:
        items (List[float, ...]): The list of ranked items.

    Returns:
        float: The DCG score.
    """
    return sum([s/np.log2(i+2) for i, s in enumerate(items)])


def ndcg_score(items):
    """Calculate the normalized discounted cumulative gain.

    Args:
        items (List[float, ...]): The list of ranked items.

    Returns:
        float: The NDCG score.
    """
    idcg = dcg_score(sorted(items, reverse=True))
    return idcg if idcg == 0 else dcg_score(items) / idcg


def perform_groupwise_evaluation(X_test, y_test, y_pred):
    """Calculate HR@10 and NDCG@10 by user.

    Args:
        X_test (ndarray of shape (n_samples, 2)): An array where each row
                                                  consists of a user and an
                                                  item.
        y_test (ndarray of shape (n_samples,)): An array where each entry
                                                denotes interactions between
                                                the corresponding user and
                                                item.
        y_pred (ndarray of shape (n_samples,)): An array where each entry
                                                denotes interactions between
                                                the corresponding user and
                                                item.

    Returns:
        Dict[str, float]: The HR@10 and NDCG@10.
    """

    # TODO: add kwdarg for groups
    # TODO: add kwdarg for metrics to return

    hits = np.empty(0)
    gains = np.empty(0)

    for user in np.unique(X_test[:, 0]):
        user_idx = X_test[:, 0] == user
        y_test_temp = y_test[user_idx]
        y_pred_temp = y_pred[user_idx]
        idx_top = np.argsort(y_pred_temp)[:-11:-1]
        interactions_top = y_test_temp[idx_top]
        hits = np.append(hits, interactions_top.sum())
        gains = np.append(gains, ndcg_score(interactions_top))

    return {"hit_ratio": hits.mean(),
            "normalized_discounted_cumulative_gain": gains.mean()}
