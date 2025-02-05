import numpy as np
import torch

from scipy.spatial.distance import jensenshannon


def disagreement(pred1: list, pred2: list):
    """
    Computes the disagreement between two lists of model predictions containing the predicted class labels.

    :param pred1: The first list of predictions.
    :param pred2: The second list of predictions.
    :return: The computed disagreement.
    """
    assert len(pred1) == len(pred2)
    return 1 / len(pred1) * len([1 for (i, j) in zip(pred1, pred2) if i != j])


def normalized_disagreement(pred1: list, pred2: list, labels: list):
    """
    Calculates the error-corrected normalized disagreement between two lists of model predictions and the ground truth
    labels.

    :param pred1: The first list of predictions.
    :param pred2: The second list of predictions.
    :param labels: The ground truth labels.
    :return: The computed disagreement.
    """
    err1 = disagreement(pred1, labels)
    err2 = disagreement(pred2, labels)
    min_dis = abs(err1 - err2)
    max_dis = min(err1 + err2, 1)
    return (disagreement(pred1, pred2) - min_dis) / (max_dis - min_dis)


def kappa(pred1: list, pred2: list, num_classes: int):
    """
    Calculates Cohen's kappa between two lists of model predictions for the given number of possible classes.

    :param pred1: The first list of predictions.
    :param pred2: The second list of predictions.
    :param num_classes: The number of possible classes.
    :return: The computed kappa value.
    """
    assert len(pred1) == len(pred2)
    assert num_classes > 0
    class_sum = 0
    for num in range(num_classes):
        sum1 = len([1 for c in pred1 if c == num])
        sum2 = len([1 for c in pred2 if c == num])
        class_sum += sum1 * sum2
    pe = (1 / (len(pred1) ** 2)) * class_sum
    po = 1 - disagreement(pred1, pred2)
    return (po - pe) / (1 - pe)


def jensen_shannon(probs1: np.ndarray, probs2: np.ndarray):
    """
    Calculates the Jensen-Shannon divergence between two arrays containing probability predictions.

    :param probs1: The first array.
    :param probs2: The second array.
    :return: The calculated JSD.
    """
    assert probs1.shape == probs2.shape, f"{probs1.shape=}, {probs2.shape=}"
    js_distance = jensenshannon(probs1, probs2, axis=1) ** 2
    np.nan_to_num(js_distance, copy=False, nan=0.0)
    js_sum = np.sum(js_distance)
    return js_sum / len(probs1)


def surrogate_churn(probs1: np.ndarray, probs2: np.ndarray, alpha: float = 1.):
    """
    Calculates the surrogate churn between two arrays containing probability predictions.

    :param probs1: The first array.
    :param probs2: The second array.
    :param alpha: The alpha value to use.
    :return: The calculated surrogate churn.
    """
    assert probs1.shape == probs2.shape
    arg1 = (probs1 / np.max(probs1, axis=1, keepdims=True)) ** alpha
    arg2 = (probs2 / np.max(probs2, axis=1, keepdims=True)) ** alpha
    argssum = np.sum(abs(arg1 - arg2))
    return argssum / (2 * len(probs1))


def ambiguity(preds: dict):
    """
    Calculates the ambiguity between model predictions passed in the given dictionary.

    :param preds: A dictionary containing one entry of predictions made by each model.
    :return: The calculated ambiguity.
    """
    assert bool(preds)
    assert len(preds[0]) > 0
    ambiguous_set = set()
    for entry1 in preds:
        for entry2 in preds:
            assert len(preds[entry1]) == len(preds[entry2])
            conflicts = np.not_equal(preds[entry1], preds[entry2])
            ambiguous_set.update(np.where(conflicts)[0])
    return len(ambiguous_set) / len(preds[0])


def discrepancy(preds: dict):
    """
    Calculates the discrepancy, i.e. the maximum disagreement of a pair of classifiers from the given set of model
    predictions.

    :param preds: A dictionary containing one entry of predictions made by each model.
    :return: The calculated discrepancy.
    """
    assert bool(preds)
    assert len(preds[0]) > 0
    num_max_discrepancy = 0
    for entry1 in preds:
        for entry2 in preds:
            assert len(preds[entry1]) == len(preds[entry2])
            conflicts = torch.not_equal(preds[entry1], preds[entry2])
            discrepancy = torch.sum(conflicts).numpy()
            num_max_discrepancy = max(discrepancy, num_max_discrepancy)
    return num_max_discrepancy / len(preds[0])
