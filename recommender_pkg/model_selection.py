import numpy as np
import pandas as pd
import random


def LeaveMembersOut(*lists, groups=None, n_val=1, n_test=1, seed=None):
    """Returns indices to split data into train, val, and test sets.

    Returns indices of train, test, and validation sets based on the given
    number of validation and test items per group. The function accepts a
    variable number of lists, which is included for consistency with similar
    `scikit-learn` functions. The length of the lists is used to determine the
    number of indices. If a list of groups is specified, the specified number
    of members of each group will be placed in the validation and test sets.

    Args:
        *lists: One or more lists from which to leave members out
        groups: A list by which the indices may be grouped (e.g. user ids)
        n_val: The number of members to be left out for the val set
        n_test: The number of members to be left out for the test set
        seed: A random seed

    Returns:
        A tuple containing lists of indices in the order train, val, test
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    if len(lists) == 0:
        raise ValueError("At least one list required as input")
    if groups is None:
        groups = [0] * len(lists[0])
    for arr in lists:
        if len(arr) != len(groups):
            raise ValueError("Groups and lists should be the same length")

    df = pd.DataFrame(groups).T.reset_index(drop=True).T
    idx_train = []
    idx_val = []
    idx_test = []
    min_len = n_test + n_val + 1

    for _, group in df.groupby(0):
        if len(group) < min_len:
            print(group)
            raise ValueError(("Each group must have at least "
                              f"{min_len} members"))
        shuffled = group.sample(frac=1)
        idx_train.extend(shuffled.iloc[(n_val+n_test):].index.to_list())
        idx_val.extend(shuffled.iloc[:n_val].index.to_list())
        idx_test.extend(shuffled.iloc[n_val:(n_val+n_test)].index.to_list())

    idx_train = random.sample(idx_train, len(idx_train))
    idx_val = random.sample(idx_val, len(idx_val))
    idx_test = random.sample(idx_test, len(idx_test))

    return idx_train, idx_val, idx_test


def sample_n_non_interactions(X, y, user_id, n=100, seed=None):
    """Samples non-interactions for a given user.

    Returns X (user, item) and y (zeros) np arrays of N (100 by default) items
    which the user has not interacted with.

    Args:
        X: An array of (user, item)
        y: An array denoting interactions
        user_id: The unique identifier of a user
        n: The number of non-interactions to sample
        seed: A random seed

    Returns:
        A tuple consisting of an array of (user, item) and an array of zeros

    """
    if seed is not None:
        np.random.seed(seed)
    items = X.iloc[:, 1].unique()
    items_i = X[X.iloc[:, 0] == user_id].iloc[:, 1]
    items_ni = np.random.choice(np.setdiff1d(items, items_i, True), n)

    return np.column_stack((np.full(n, user_id), items_ni)), np.zeros(n)
