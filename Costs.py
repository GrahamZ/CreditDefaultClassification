"""Binary classification cost curve given a fixed cost matrix.

Adapted from sklearn _binary_clf_curve.

"""

# Author: Graham Hay


import numpy as np


def cost_curve(y_true, y_score, cost_matrix):
    """Compute the cost curve given a cost matrix.

    Note: this implementation is restricted to the binary classification task.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True binary labels:  {0, 1}

    y_score : ndarray of shape (n_samples,)
        Target scores: probability estimates of the positive class.

    cost_matrix : ndarray of shape (2, 2)
        The mis-classification costs defined as:
        [Cost of TN, Cost of FP]
        [Cost of FN, Cost of TP]

    Returns
    -------

    tc : ndarray of shape (>2,)
        Total cost of mis-classification such that element i is calculated
        from positive predictions with score >= `thresholds[i]`.

    thresholds : ndarray of shape = (n_thresholds,)
        Decreasing thresholds on the decision function used to compute
        fpc and fnc. `thresholds[0]` represents no instances being predicted
        and is arbitrarily set to `max(y_score) + 1`.

"""
    # make y_true a boolean vector
    y_true = (y_true == 1)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    p = np.sum(y_true)
    n = y_true.size - p
    fns = p - tps
    tns = n - fps

    thresholds = y_score[threshold_idxs]
    if y_score[threshold_idxs][0] < 1.0:
        thresholds = np.r_[1.0, thresholds]
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        fns = np.r_[p, fns]
        tns = np.r_[n, tns]

    if y_score[threshold_idxs][-1] > 0.0:
        thresholds = np.r_[thresholds, 0.0]
        tps = np.r_[tps, p]
        fps = np.r_[fps, n]
        fns = np.r_[fns, 0]
        tns = np.r_[tns, 0]

    tcs = tps*cost_matrix[1, 1] + fps*cost_matrix[0, 1] + fns*cost_matrix[1, 0] + tns*cost_matrix[0, 0]

    return tcs, thresholds


def expected_cost(y_true, y_score, cost_matrix):
    """Compute the expected cost given a cost matrix.

    Note: this implementation is restricted to the binary classification task.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True binary labels:  {0, 1}

    y_score : ndarray of shape (n_samples,)
        Target scores: probability estimates of the positive class.

    cost_matrix : ndarray of shape (2, 2)
        The mis-classification costs defined as:
        [Cost of TN, Cost of FP]
        [Cost of FN, Cost of TP]

    Returns
    -------

    tc : scalar
        Expected value of the cost

"""
    cost_of_positives = y_true * (y_score * cost_matrix[1,1] + (1.0 - y_score) * cost_matrix[1,0])
    cost_of_negatives = (1.0 - y_true) * (y_score * cost_matrix[0,1] + (1.0 - y_score) * cost_matrix[0,0])
    costs = cost_of_positives + cost_of_negatives
    return costs.sum()

