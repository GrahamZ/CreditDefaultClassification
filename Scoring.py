"""Determine the threshold in binary classification.
   - for a given recall
   - for minimum cost
"""

# Author: Graham Hay


import numpy as np
from sklearn.metrics import roc_curve
from Costs import cost_curve


class RecallThreshold:
    """For a binary classifier, determine the threshold yielding a recall that meets or exceeds a target recall.

    Implemented as a callable compatible with sklearn.
    """
    def __init__(self, target_recall):
        """Constructor

        :param target_recall: target for recall to be met or exceeded
        """
        self.tolerance = 0.0001
        self.target_recall = target_recall

    def __call__(self, estimator, X, y):
        y_probs = estimator.predict_proba(X)[:,1]
        fprs, tprs, thresholds = roc_curve(y, y_probs)
        idx = np.argmax(tprs >= self.target_recall)
        if idx > 0 and tprs[idx] - tprs[idx-1] > self.tolerance:
            alpha = (self.target_recall - tprs[idx-1])/(tprs[idx] - tprs[idx-1])
            t = thresholds[idx-1]*(1.0 - alpha) + thresholds[idx]*alpha
            return t
        else:
            return thresholds[idx]


class CostThreshold:
    """For a binary classifier, determine the threshold yielding the minimum cost.

    Implemented as a callable compatible with sklearn.
    """
    def __init__(self, cost_matrix):
        """Constructor

        :param cost_matrix: ndarray of shape (2, 2)
        """
        self.cost_matrix = cost_matrix

    def __call__(self, estimator, X, y):
        y_probs = estimator.predict_proba(X)[:,1]
        costs, thresholds = cost_curve(y, y_probs, self.cost_matrix)
        idx = np.argmin(costs)
        return thresholds[idx]

