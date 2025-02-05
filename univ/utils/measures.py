import gc
import numpy as np
import pandas as pd
import torch
import os

import rtd
from univ.measures import funct_sim
from univ.measures import rep_sim as rs
from random import shuffle
from torch.nn.functional import pad
from tqdm import tqdm
from loguru import logger


def get_class_predictions(outputs: dict):
    """
    Computes the predicted class labels by taking the argmax of the outputs passed in the given dictionary. Each entry
    is expected to contain the outputs for a specific model.

    :param outputs: The dictionary containing the outputs of each model.
    :return: A dictionary containing the classes predicted by each model.
    """
    preds = {}
    for entry in outputs:
        preds[entry] = torch.argmax(outputs[entry], dim=1)
    return preds


def get_probabilities(outputs: dict):
    """
    Computes the predicted probabilities using softmax for each entry passed in the given dictionary. Each entry is
    expected to contain the outputs for a specific model.

    :param outputs: The dictionary containing the outputs of each model.
    :return: A dictionary containing the probabilities predicted by each model.
    """
    probs = {}
    for entry in outputs:
        probs[entry] = torch.nn.functional.softmax(outputs[entry], dim=1).numpy()
    return probs


def get_predictions_inverted(outputs: dict):
    """
    Computes the predicted class labels by taking the argmax of the outputs passed in the given dictionary. Each entry
    is expected to contain another dictionary containing one entry per model with the outputs computed on a specific set
    on inverted images.

    :param outputs: The dictionary containing the model outputs.
    :return: A dictionary containing the classes predicted by each model.
    """
    preds = {}
    for entry in outputs:
        preds[entry] = get_class_predictions(outputs[entry])
    return preds


def get_probabilities_inverted(outputs: dict):
    """
    Computes the predicted probabilities of the outputs passed in the given dictionary. Each entry is expected to
    contain another dictionary containing one entry per model with the outputs computed on a specific set on inverted
    images.

    :param outputs: The dictionary containing the model outputs.
    :return: A dictionary containing the probabilities predicted by each model.
    """
    probs = {}
    for entry in outputs:
        probs[entry] = get_probabilities(outputs[entry])
    return probs


def compute_funct_sim(outputs1: any, outputs2: any, measure: str, labels: list = None, num_classes: int = 0,
                      alpha: float = 1.0):
    """
    Computes different functional similarity measures between the two given outputs, which can either be the predicted
    class labels or probabilities.

    :param outputs1: The outputs of the first model.
    :param outputs2: The outputs of the second model.
    :param measure: The measure to use for comparison.
    :param labels: A list of ground truth labels used to calculate normalized disagreement, default: None.
    :param num_classes: The total number of classes used to calculate Cohen's kappa, default: 0.
    :param alpha: The alpha value to use when computing surrogate churn, default: 1.0.
    :return: The calculated similarity score.
    """
    if measure == 'dis':
        return funct_sim.disagreement(outputs1.tolist(), outputs2.tolist())
    elif measure == 'norm':
        return funct_sim.normalized_disagreement(outputs1.tolist(), outputs2.tolist(), labels)
    elif measure == 'kappa':
        return funct_sim.kappa(outputs1.tolist(), outputs2.tolist(), num_classes)
    elif measure == 'jsd':
        return funct_sim.jensen_shannon(outputs1, outputs2)
    else:
        return funct_sim.surrogate_churn(outputs1, outputs2, alpha)


def compute_permuted_funct_sim(outputs1: any, outputs2: any, measure: str, labels: list = None, num_classes: int = 0,
                               alpha: float = 1.0, n_permutations: int = 10):
    """
    Randomly permutes the given second model outputs and computes the desired functional similarity score between the
    permuted values and the other outputs. This process is repeated according to the given number of permutations and
    the average score is returned.

    :param outputs1: The outputs of the first model.
    :param outputs2: The outputs of the second model.
    :param measure: The measure to use for comparison.
    :param labels: A list of ground truth labels used to calculate normalized disagreement, default: None.
    :param num_classes: The total number of classes used to calculate Cohen's kappa, default: 0.
    :param alpha: The alpha value to use when computing surrogate churn, default: 1.0.
    :param n_permutations: The number of random permutations to compute for an average result, default: 10.
    :return: The average of the computed scores.
    """
    scores = []
    for i in range(n_permutations):
        if isinstance(outputs2, np.ndarray):
            permuted = np.arange(outputs2.shape[0])
            np.random.shuffle(permuted)
        else:
            permuted = torch.randperm(outputs2.size(0))
        scores.append(compute_funct_sim(outputs1, outputs2[permuted], measure, labels, num_classes, alpha))
    return np.mean(scores), scores


def get_funct_sim(outputs: dict, measure: str, labels: list = None, num_classes: int = 0, alpha: float = 1.0,
                  permute: bool = False, n_permutations: int = 10):
    """
    Computes the desired functional similarity scores between model outputs contained in the given dictionary.

    :param outputs: The dictionary containing the model predictions.
    :param measure: The measure to use for comparison.
    :param labels: A list of ground truth labels used to calculate normalized disagreement, default: None.
    :param num_classes: The total number of classes used to calculate Cohen's kappa, default: 10.
    :param alpha: The alpha value to use when computing surrogate churn, default: 1.0.
    :param permute: Whether to compute a baseline score on permuted inputs, default: False.
    :param n_permutations: The number of random permutations to compute for an average result, default: 10.
    :return: The calculated similarity scores.
    """
    sim, base = [], []
    for entry1 in outputs:
        for entry2 in outputs:
            if permute:
                mean_scores, scores = compute_permuted_funct_sim(outputs[entry1], outputs[entry2], measure, labels,
                                                                 num_classes, alpha, n_permutations)
                sim.append(mean_scores)
                base.append(scores)
            else:
                sim.append(compute_funct_sim(outputs[entry1], outputs[entry2], measure, labels, num_classes, alpha))
    return sim, base


def get_inverted_funct_sim(outputs: dict, measure: str, labels: list = None, num_classes: int = 0, alpha: float = 1.0,
                           permute: bool = False, n_permutations: int = 10):
    """
    Computes the desired functional similarity scores between model outputs contained in the given dictionary calculated
    on inverted images. Each entry of the dictionary should contain another dictionary with outputs for all models
    calculated on a set of inverted images generated for a specific model.

    :param outputs: The dictionary containing the model predictions.
    :param measure: The measure to use for comparison.
    :param labels: A list of ground truth labels used to calculate normalized disagreement, default: None.
    :param num_classes: The total number of classes used to calculate Cohen's kappa, default: 0.
    :param alpha: The alpha value to use when computing surrogate churn, default: 1.0.
    :param permute: Whether to compute a baseline score on permuted inputs, default: False.
    :param n_permutations: The number of random permutations to compute for an average result, default: 10.
    :return: The calculated similarity scores.
    """
    sim, base = [], []
    for entry1 in outputs:
        row_output = outputs[entry1][entry1]
        if len(labels) == len(outputs):
            y = labels[entry1]
        else:
            y = labels
        for entry2 in outputs[entry1]:
            logger.debug(f"Comparing {entry1}, {entry2}")
            if permute:
                mean_scores, scores = compute_permuted_funct_sim(row_output, outputs[entry1][entry2], measure, y,
                                                                 num_classes, alpha, n_permutations)
                sim.append(mean_scores)
                base.append(scores)
            else:
                sim.append(compute_funct_sim(row_output, outputs[entry1][entry2], measure, y, num_classes, alpha))
    return sim, base


def get_inverted_statistics(outputs: dict, measure: str):
    """
    Computes ambiguity or discrepancy between model outputs contained in the given dictionary calculated on inverted
    images. Each entry of the dictionary should contain another dictionary with outputs for all models calculated on a
    set of inverted images generated for a specific model.

    :param outputs: The dictionary containing the model predictions.
    :param measure: The measure to use for comparison.
    :return: The calculated scores for each entry contained in the dictionary.
    """
    stats = []
    for entry in outputs:
        if measure == 'amb':
            stats.append(funct_sim.ambiguity(outputs[entry]))
        else:
            stats.append(funct_sim.discrepancy(outputs[entry]))
    return stats


def get_self_similarity(activations: dict, measure: str, entry1: int):
    """
    Returns the similarity score of a set of activations with itself depending on the given similarity measure.

    :param activations: The dictionary containing the model activations.
    :param measure: The measure to use for comparison.
    :param entry1: The number of the current entry used to index the dictionary.
    :return: 0 for procrustes and the shift shape metric, 1 for everything else.
    """
    if measure in ['proc', 'gulp', 'rtd']:
        return 0
    elif measure == 'shape':
        if len(activations[entry1][entry1].shape) == 4:
            return 0
    else:
        return 1


def compute_rep_sim(
        activations1: list,
        activations2: list,
        device: torch.device,
        eps: float = 1e-8,
        alpha: float = 1.0,
        center_columns: bool = True,
        k: int = 500,
        sim_funct: str = 'euc',
        measure: str = 'cos',
        gulp_lambda: float = 0,
        use_norm: bool = True,
):
    """
    Computes different representational similarity measures between the two given lists of model activations.

    :param activations1: The activations of the first model.
    :param activations2: The activations of the second model.
    :param device: The device to which the activations should be transferred.
    :param eps: Small number to avoid zero-division in cosine similarity computation, default: 1e-8.
    :param alpha: Regularization parameter for the shift shape metric, default: 1.0.
    :param center_columns: Boolean indicating whether columns should be mean-centered, default: True.
    :param k: The number of nearest neighbors to consider, default: 500.
    :param sim_funct: The similarity function to use for nearest neighbor calculation, default: 'euc'.
    :param measure: The measure to use for comparison, default: 'cos'.
    :param use_norm: Boolean indicating whether the activation matrices should be normalized for Procrustes.
    :return: The computed similarity score.
    """
    rep1 = torch.flatten(torch.vstack(activations1), start_dim=1).to(device)
    rep2 = torch.flatten(torch.vstack(activations2), start_dim=1).to(device)
    logger.debug(f"Comparing reps with {rep1.shape=}, {rep2.shape=}")
    if measure == 'cos':
        return rs.second_order_cos_sim(rep1, rep2, center_columns, eps).cpu()
    elif measure == 'jacc':
        return torch.mean(rs.knn_jaccard(rep1, rep2, center_columns, k, sim_funct)).cpu()
    elif measure == 'rank':
        return torch.mean(rs.rank_similarity(rep1, rep2, center_columns, k, sim_funct)).cpu()
    elif measure == 'rank_jacc':
        return rs.joint_rank_jaccard(rep1, rep2, center_columns, k, sim_funct).cpu()
    elif measure == 'proc':
        return rs.orthogonal_procrustes(rep1, rep2, center_columns, use_norm).cpu()
    elif measure == 'gulp':
        return rs.gulp(rep1, rep2, lmbda=gulp_lambda)
    elif measure == 'rtd':
        return rtd.rtd(rep1, rep2)
    elif measure == 'shape':
        if len(activations1[0].shape) == len(activations2[0].shape) == 4:
            return rs.compute_shift_shape(torch.vstack(activations1), torch.vstack(activations2), alpha, center_columns)
    else:
        raise ValueError('Invalid measure provided!')


def compute_permuted_rep_sim(activations1: list, activations2: list, device: torch.device, eps: float = 1e-8,
                             alpha: float = 1.0, center_columns: bool = True, k: int = 500, sim_funct: str = 'euc',
                             measure: str = 'cos', n_permutations: int = 10, use_norm: bool = True,
                             gulp_lambda: float = 0):
    """
    Randomly permutes the given second model activations and computes the desired representational similarity score
    between the permuted values and the other activations. This process is repeated according to the given number of
    permutations and the average score is returned.

    :param activations1: The activations of the first model.
    :param activations2: The activations of the second model.
    :param device: The device to which the activations should be transferred.
    :param eps: Small number to avoid zero-division in cosine similarity computation, default: 1e-8.
    :param alpha: Regularization parameter for the shift shape metric, default: 1.0.
    :param center_columns: Boolean indicating whether columns should be mean-centered, default: True.
    :param k: The number of nearest neighbors to consider, default: 500.
    :param sim_funct: The similarity function to use for nearest neighbor calculation, default: 'euc'.
    :param measure: The measure to use for comparison, default: 'cos'.
    :param n_permutations: The number of random permutations to compute for an average result, default: 10.
    :param use_norm: Boolean indicating whether the activation matrices should be normalized for Procrustes.
    :return: The average of the computed scores and all computed scores.
    """
    scores = []
    for i in range(n_permutations):
        permuted = activations2.copy()
        shuffle(permuted)
        scores.append(
            compute_rep_sim(
                activations1=activations1,
                activations2=permuted,
                device=device,
                eps=eps,
                alpha=alpha,
                center_columns=center_columns,
                k=k,
                sim_funct=sim_funct,
                measure=measure,
                use_norm=use_norm,
                gulp_lambda=gulp_lambda,
            )
        )
    if None not in scores:
        return np.mean(scores), scores
    else:
        return None, None


def get_rep_sim(activations: dict, device: torch.device, eps: float = 1e-8, alpha: float = 1.0,
                center_columns: bool = True, k: int = 500, sim_funct: str = 'euc', measure: str = 'cos',
                permute: bool = False, n_permutations: int = 10, use_norm: bool = True, gulp_lambda: float = 0):
    """
    Computes the desired representational similarity scores between model activations contained in the given dictionary.
    Each entry in the dictionary is expected to contain a tensor of activations for a specific model.

    :param activations: The dictionary containing the model activations.
    :param device: The device to which the activations should be transferred.
    :param eps: Small number to avoid zero-division in cosine similarity computation, default: 1e-8.
    :param alpha: Regularization parameter for the shift shape metric, default: 1.0.
    :param center_columns: Boolean indicating whether columns should be mean-centered, default: True.
    :param k: The number of nearest neighbors to consider, default: 500.
    :param sim_funct: The similarity function to use for nearest neighbor calculation, default: 'euc'.
    :param measure: The measure to use for comparison, default: 'cos'.
    :param permute: Whether to compute a baseline score on permuted inputs, default: False.
    :param n_permutations: The number of random permutations to compute for an average result, default: 10.
    :param use_norm: Boolean indicating whether the activation matrices should be normalized for Procrustes.
    :return: The calculated similarity scores.
    """
    sim, base = [], []
    for entry1 in activations:
        for entry2 in tqdm(activations, desc='Similarity for model {}'.format(entry1), total=len(activations)):
            if permute:
                mean_scores, scores = compute_permuted_rep_sim(activations[entry1], activations[entry2], device, eps,
                                                               alpha, center_columns, k, sim_funct, measure,
                                                               n_permutations, use_norm)
                sim.append(mean_scores)
                base.append(scores)
            else:
                if entry1 == entry2:
                    sim.append(get_self_similarity(activations, measure, entry1))
                elif entry1 > entry2:
                    sim.append(sim[entry2 * len(activations) + entry1])
                else:
                    sim.append(
                        compute_rep_sim(
                            activations1=activations[entry1],
                            activations2=activations[entry2],
                            device=device,
                            eps=eps,
                            alpha=alpha,
                            center_columns=center_columns,
                            k=k,
                            sim_funct=sim_funct,
                            measure=measure,
                            use_norm=use_norm,
                            gulp_lambda=gulp_lambda,
                        )
                    )
    torch.cuda.empty_cache()
    return sim, base


def get_inverted_rep_sim(activations: dict, device: torch.device, eps: float = 1e-8, alpha: float = 1.0,
                         center_columns: bool = True, k: int = 500, sim_funct: str = 'euc', measure: str = 'cos',
                         permute: bool = False, n_permutations: int = 10, use_norm: bool = True, gulp_lambda: float = 0):
    """
    Computes the desired representational similarity scores between model outputs contained in the given dictionary
    calculated on inverted images. Each entry of the dictionary should contain another dictionary with outputs for all
    models calculated on a set of inverted images generated for a specific model.

    :param activations: The dictionary containing the model activations.
    :param device: The device to which the activations should be transferred.
    :param eps: Small number to avoid zero-division in cosine similarity computation, default: 1e-8.
    :param alpha: Regularization parameter for the shift shape metric, default: 1.0.
    :param center_columns: Boolean indicating whether columns should be mean-centered, default: True.
    :param k: The number of nearest neighbors to consider, default: 500.
    :param sim_funct: The similarity function to use for nearest neighbor calculation, default: 'euc'.
    :param measure: The measure to use for comparison, default: 'cos'.
    :param permute: Whether to compute a baseline score on permuted inputs, default: False.
    :param n_permutations: The number of random permutations to compute for an average result, default: 10.
    :param use_norm: Boolean indicating whether the activation matrices should be normalized for Procrustes.
    :return: The calculated similarity scores.
    """
    sim, base = [], []
    for entry1 in activations:
        row_output = activations[entry1][entry1]
        for entry2 in tqdm(activations[entry1], desc='Similarity for model {}'.format(entry1), total=len(activations)):
            if permute:
                mean_scores, scores = compute_permuted_rep_sim(
                    row_output,
                    activations[entry1][entry2],
                    device=device,
                    eps=eps,
                    alpha=alpha,
                    center_columns=center_columns,
                    k=k,
                    sim_funct=sim_funct,
                    measure=measure,
                    n_permutations=n_permutations,
                    use_norm=use_norm,
                    gulp_lambda=gulp_lambda,
                )
                sim.append(mean_scores)
                base.append(scores)
            else:
                if entry1 == entry2:
                    sim.append(get_self_similarity(activations[entry1], measure, entry1))
                else:
                    sim.append(
                        compute_rep_sim(
                            activations1=row_output,
                            activations2=activations[entry1][entry2],
                            device=device,
                            eps=eps,
                            alpha=alpha,
                            center_columns=center_columns,
                            k=k,
                            sim_funct=sim_funct,
                            measure=measure,
                            use_norm=use_norm,
                            gulp_lambda=gulp_lambda,
                        )
                    )
    gc.collect()
    torch.cuda.empty_cache()
    return sim, base


def compute_rep_sim_statistics(activations: any, measure: str = 'mag'):
    """
    Computes different representational similarity statistics for a specific set of activations generated by a single
    model or over a dictionary containing activations generated by different models.

    :param activations: A tensor containing activations of a single model or a dictionary containing entries with
    activations of different models.
    :param measure: The statistics measure to compute, default: 'mag'.
    :return: The computed similarity statistics.
    """
    if measure == 'mag':
        return rs.magnitude(torch.flatten(torch.vstack(activations), start_dim=1))
    elif measure == 'con':
        return rs.concentricity(torch.flatten(torch.vstack(activations), start_dim=1))
    elif measure == 'mag_var':
        return rs.magnitude_variance(activations)
    elif measure == 'con_var':
        return rs.concentricity_variance(activations)
    else:
        raise ValueError('Invalid statistics measure provided!')


def get_rep_sim_statistics(activations: dict, measure: str = 'mag'):
    """
    Computes the desired representational similarity statistics of model activations contained in the given dictionary.
    Each entry in the dictionary is expected to contain a tensor of activations for a specific model.

    :param activations: The dictionary containing the model activations.
    :param measure: The statistics measure to compute, default: 'mag'.
    :return: The computed similarity statistics.
    """
    stats = []
    for entry in activations:
        stats.append(compute_rep_sim_statistics(activations[entry], measure))
    return stats


def get_inverted_rep_sim_statistics(activations: dict, measure: str = 'mag'):
    """
    Computes the desired representational similarity statistics between model outputs contained in the given dictionary
    calculated on inverted images. Each entry of the dictionary should contain another dictionary with outputs for all
    models calculated on a set of inverted images generated for a specific model.

    :param activations: The dictionary containing the model activations.
    :param measure: The statistics measure to compute, default: 'mag'.
    :return: The computed similarity statistics.
    """
    stats = []
    for entry1 in activations:
        if measure in ['mag_var', 'con_var']:
            stats.append(compute_rep_sim_statistics(activations[entry1], measure))
        else:
            for entry2 in activations[entry1]:
                stats.append(compute_rep_sim_statistics(activations[entry1][entry2], measure))
    return stats


def check_pad_dim(dim1: int, dim2: int):
    """
    Given two numbers indicating the length of dimensions of activations along one axis, they are compared to check if
    any padding is required for the smaller number. The required number of padding rows for each case is then returned.

    :param dim1: The length of the dimension of the first model.
    :param dim2: The length of the dimension of the second model.
    :return: The required padding rows for both models.
    """
    if dim1 == dim2:
        return 0, 0
    if dim1 > dim2:
        return 0, dim1 - dim2
    if dim2 > dim1:
        return dim2 - dim1, 0


def pad_activations(activations1: torch.Tensor, activations2: torch.Tensor):
    """
    Given two four-dimensional tensors, they are padded to match in size along all four dimensions. Along each
    dimension, the tensor with the lower size is padded along this dimension with zeros.

    :param activations1: The first set of activations.
    :param activations2: The second set of activations.
    :return: The zero-padded activations.
    """
    n1, h1, w1, c1 = activations1.shape
    n2, h2, w2, c2 = activations2.shape

    pad_n1, pad_n2 = check_pad_dim(n1, n2)
    pad_h1, pad_h2 = check_pad_dim(h1, h2)
    pad_w1, pad_w2 = check_pad_dim(w1, w2)
    pad_c1, pad_c2 = check_pad_dim(c1, c2)

    acts1 = pad(activations1, (0, pad_c1, 0, pad_w1, 0, pad_h1, 0, pad_n1))
    acts2 = pad(activations2, (0, pad_c2, 0, pad_w2, 0, pad_h2, 0, pad_n2))
    return acts1, acts2


def save_dataframe_to_csv(df: pd.DataFrame, column_names: list, save_path: str):
    """
    Saves the data contained in the given dataframe as a csv file using the specified path and sets the columns names of
    the data to the ones passed in the given list.

    :param df: The dataframe containing the data to be saved.
    :param column_names: The column names to be set.
    :param save_path: The path and file name specifying the file save location.
    """
    df = df.transpose()
    df.columns = column_names
    os.makedirs(os.path.split(save_path)[0], exist_ok=True)
    df.to_csv(save_path)


def save_baseline_scores(scores: list, save_path: str, names: list, num_perturbations: int = 10):
    """
    Saves all calculated baseline scores in a single CSV file, with the scores of the individual perturbations being
    saved as a list in a cell.

    :param scores: The baseline scores.
    :param save_path: The path where the scores should be saved.
    :param names: The names of the columns.
    :param num_perturbations: The number of perturbations that were performed.
    """
    if scores:
        array = np.array(scores).reshape(len(names), len(names), num_perturbations)
        sc = array.tolist()
        df = pd.DataFrame()
        for idx, name in enumerate(names):
            df[idx] = sc[idx]
        df = df.transpose()
        df.columns = names
        df.to_csv(save_path)
