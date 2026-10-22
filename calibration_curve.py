import numpy as np
from quantiles import weighted_quantile
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_consistent_length, _check_pos_label_consistency


def weighted_calibration_curve(
    y_true,
    y_prob,
    *,
    sample_weight=None,
    pos_label=None,
    n_bins=5,
    strategy="uniform",
    return_bin_total=False
):
    """Compute true and predicted probabilities for a calibration curve.

    The method assumes the inputs come from a binary classifier, and
    discretize the [0, 1] interval into bins.

    Calibration curves may also be referred to as reliability diagrams.

    Read more in the :ref:`User Guide <calibration>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True targets.

    y_prob : array-like of shape (n_samples,)
        Probabilities of the positive class.

    pos_label : int, float, bool or str, default=None
        The label of the positive class.

        .. versionadded:: 1.1

    n_bins : int, default=5
        Number of bins to discretize the [0, 1] interval. A bigger number
        requires more data. Bins with no samples (i.e. without
        corresponding values in `y_prob`) will not be returned, thus the
        returned arrays may have less than `n_bins` values.

    strategy : {'uniform', 'quantile'}, default='uniform'
        Strategy used to define the widths of the bins.

        uniform
            The bins have identical widths.
        quantile
            The bins have the same number of samples and depend on `y_prob`.

    Returns
    -------
    prob_true : ndarray of shape (n_bins,) or smaller
        The proportion of samples whose class is the positive class, in each
        bin (fraction of positives).

    prob_pred : ndarray of shape (n_bins,) or smaller
        The mean predicted probability in each bin.

    References
    ----------
    Alexandru Niculescu-Mizil and Rich Caruana (2005) Predicting Good
    Probabilities With Supervised Learning, in Proceedings of the 22nd
    International Conference on Machine Learning (ICML).
    See section 4 (Qualitative Analysis of Predictions).

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.calibration import calibration_curve
    >>> y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    >>> y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9,  1.])
    >>> prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=3)
    >>> prob_true
    array([0. , 0.5, 1. ])
    >>> prob_pred
    array([0.2  , 0.525, 0.85 ])
    """
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)
    check_consistent_length(y_true, y_prob)
    pos_label = _check_pos_label_consistency(pos_label, y_true)

    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1].")

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError(
            f"Only binary classification is supported. Provided labels {labels}."
        )
    y_true = y_true == pos_label

    if strategy == "quantile":  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = weighted_quantile(y_prob, quantiles, sample_weight=sample_weight)
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(
            "Invalid entry to 'strategy' input. Strategy "
            "must be either 'quantile' or 'uniform'."
        )

    binids = np.searchsorted(bins[1:-1], y_prob)

    if sample_weight is None:
        bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
        bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
        bin_total = np.bincount(binids, minlength=len(bins))
    else:
        bin_sums = np.bincount(binids, weights=y_prob * sample_weight, minlength=len(bins))
        bin_true = np.bincount(binids, weights=y_true * sample_weight, minlength=len(bins))
        bin_total = np.bincount(binids, weights=sample_weight, minlength=len(bins))

    nonzero = bin_total != 0
    # There is an issue with how the bins are computed up there, which necessarily adds
    # an empty bin at the end. The initial code removes the zero length bins, but this
    # may shift the bins.
    # prob_true = bin_true[nonzero] / bin_total[nonzero]
    # prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    prob_true = np.divide(bin_true, bin_total, out=np.ones(n_bins+1)*np.nan, where=nonzero)
    prob_pred = np.divide(bin_sums, bin_total, out=np.ones(n_bins+1)*np.nan, where=nonzero)

    if return_bin_total:
        return prob_true, prob_pred, bin_total
    else:
        return prob_true, prob_pred
