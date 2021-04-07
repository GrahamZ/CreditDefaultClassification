"""Plotting utilities

"""

# Author: Graham Hay


import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve


def plot_calibration_curve(y_true, y_probs, n_bins=10, ax=None):
    """Plot the probability calibration curve.

    :param y_true: ndarray of shape (n_samples,)
        True binary labels:  {0, 1}
    :param y_probs: ndarray of shape (n_samples,)
        Target probabilities: probability estimates of the positive class.
    :param n_bins: integer
        Number of bins in the histogram
    :param ax: matplotlib axes
        Axes object to plot on. If None, the current axes are used.
    :return: None
    """
    if ax is None:
        ax = plt.gca()
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=n_bins)
    weights = np.ones_like(y_probs) / y_probs.size
    ax.hist(y_probs, weights=weights, range=(0, 1), bins=n_bins, histtype='bar',
            color='g', alpha=0.5, lw=2)
    ax.plot([0, 1], [0, 1], "k:", label="Ideal curve")
    ax.plot(prob_pred, prob_true, "s-")

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set(aspect='equal')


def plot_cost_curve(costs, thresholds, ax=None):
    """Plot the cost curve with the minimum cost threshold.

    :param costs: ndarray of shape (n_samples,)
        Costs corresponding to the thresholds
    :param thresholds: ndarray of shape (n_samples,)
        Thresholds
    :param ax: matplotlib axes
        Axes object to plot on. If None, the current axes are used.
    :return: None
    """
    if ax is None:
        ax = plt.gca()
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Cost')
    ax.set_xlim([0.0, 1.0])
    ax.grid(True)
    ax.plot(thresholds, costs)
    # Plot the optimal threshold
    t = thresholds[np.argmin(costs)]
    cost = costs[np.argmin(costs)]
    ax.plot(t, cost, '+', color='orange')
    ax.annotate('Threshold={:5.3f}'.format(t), xy=(t, cost), xytext=(20, 0), textcoords='offset points')
