import numpy as np
import multiprocessing
from itertools import islice
from multiprocessing import Pool

# Rashomon check on loss
def rashomon_check(cand_loss: float, ref_loss: float, epsilon: float) -> bool:
    """
    Check if the candidate model is within the Rashomon set.
    Args:
        cand_loss (float): The log loss of the candidate model.
        ref_loss (float): The log loss of the reference model.
        epsilon (float): The Rashomon parameter
    Returns:
        bool: True if the candidate model is within the Rashomon set, False otherwise.
    """
    return cand_loss <= ref_loss + epsilon


# Soft-disagreement
def total_variation_distance(P: np.ndarray, Q: np.ndarray, epsilon=1e-12) -> np.ndarray:
    """
    Compute the Total Variation Distance (TVD) between two batches of probability distributions.

    Args:
    ----------
    P, Q : (n_samples, n_classes) arrays
        Batch of probability vectors (e.g., softmax outputs).
    epsilon : float
        Small value to ensure numerical stability and avoid division by 0.

    Returns
    -------
    tvd : (n_samples,) ndarray
        TVD value per row pair in the batch. Range: [0, 1]
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    # Ensure proper probability distributions
    P = P / (P.sum(axis=1, keepdims=True) + epsilon)
    Q = Q / (Q.sum(axis=1, keepdims=True) + epsilon)

    tvd = 0.5 * np.sum(np.abs(P - Q), axis=1)
    return tvd


# Decision-based metrics for evaluating Rashomon sets
def ambiguity(rashomon_labels: np.ndarray, reference_labels: np.ndarray) -> float:
    """
        Probability that at least one Rashomon model predicts a label
        different from the reference model (or ground truth).

        Args:
        ----------
        decisions : (n_models, n_samples) int
            Hard labels from each candidate model.
        y_ref     : (n_samples,) int
            Baseline predictions OR true labels.

        Returns
        -------
        float  in [0, 1]
    """
    disagree = rashomon_labels != reference_labels.reshape(1, -1)
    return np.any(disagree, axis=0).mean()


def discrepancy(rashomon_labels: np.ndarray, reference_labels: np.ndarray) -> float:
    """
        Worst-case fraction of samples whose prediction could change
        if we swapped the deployed model for *some* model in the Rashomon set.

        Same definition as Marx et al. (ICML 2020).

        Args:
        ----------
        decisions : (n_models, n_samples) int
        y_ref     : (n_samples,) int

        Returns
        -------
        float  in [0, 1]
    """
    per_model_change = (rashomon_labels != reference_labels.reshape(1, -1)).mean(axis=1)
    return per_model_change.max()



def true_class_probablities(probabilities: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Extract P(y_true) per model & sample (vectorized).

    Args
    ----
    probabilities : (M, N, C) array
        Per-model, per-sample, per-class probabilities.
    y : (N,) int array
        True labels in [0, C).

    Returns
    -------
    (M, N) array where out[m, n] = P_model_m(y[n] | x[n]).
    """
    probabilities = np.asarray(probabilities)
    y = np.asarray(y)
    # Vectorized gather along class axis
    return np.take_along_axis(probabilities, y[None, :, None], axis=2)[:, :, 0]


def viable_prediction_range(rashomon_preds_for_truth_label: np.ndarray) -> tuple:
    """
    Compute per-sample viable prediction ranges over models.

    Args
    ----
    rashomon_preds_for_truth_label : (M, N) array
        True-class probabilities per model (rows) and sample (cols).

    Returns
    -------
    mins   : (N,) per-sample min across models
    maxs   : (N,) per-sample max across models
    widths : (N,) per-sample width = max - min
    """
    mins = rashomon_preds_for_truth_label.min(axis=0)
    maxs = rashomon_preds_for_truth_label.max(axis=0)
    return mins, maxs, maxs - mins

# Taken from the dropout paper: https://arxiv.org/abs/2402.00728

def blahut_arimoto(Pygw, log_base=2, epsilon=1e-12, max_iter=1e3):
    """
    Pygw: (m, c) matrix of class probabilities per model for ONE sample.
    Returns: capacity (float), r (optimal Pw), loop count
    """
    m, c = Pygw.shape
    Pw = np.ones((m,)) / m
    for cnt in range(int(max_iter)):
        q = (Pw * Pygw.T).T
        q = q / q.sum(axis=0)

        r = np.prod(np.power(q, Pygw), axis=1)
        r = r / r.sum()

        if np.sum((r - Pw) ** 2) / m < epsilon:
            break
        else:
            Pw = r

    capacity = 0.0
    for i in range(m):
        for j in range(c):
            if r[i] > 0 and q[i, j] > 0:  # no extra epsilons; match original
                capacity += r[i] * Pygw[i, j] * np.log(q[i, j] / r[i])
    capacity = capacity / np.log(log_base)
    return capacity, r, cnt + 1

# Taken from the dropout paper: https://arxiv.org/abs/2402.00728

def rashomon_capacity(scores_MNC):
    """
    scores_MNC: (M, N, C) probability tensor over models, samples, classes.
    Returns: (N,) per-sample capacities (bits).
    Parallelized over samples exactly like the reference.
    """
    nmodel, nsample, nclass = scores_MNC.shape
    cores = multiprocessing.cpu_count() - 1
    it = iter(range(nsample))
    ln = list(iter(lambda: tuple(islice(it, 1)), ()))  # list of singleton indices
    with Pool(cores) as p:
        cvals = p.map(
            blahut_arimoto,
            [scores_MNC[:, ix, :].reshape((nmodel, nclass)) for ix in ln],
        )
    capacity = np.array([v[0] for v in cvals])
    return capacity