import numpy as np
import torch
import math

from univ.utils import measures as ms
from univ.measures.shift_shape import LinearMetric
from torch.nn.functional import cosine_similarity, pad
from typing import Literal

# Code taken from https://haydn.fgl.dev/posts/a-better-index-of-similarity/
def cka(A: torch.Tensor, B: torch.Tensor):
    # Mean center each neuron
    A = A - torch.mean(A, dim=0, keepdim=True)
    B = B - torch.mean(B, dim=0, keepdim=True)

    dot_product_similarity = torch.linalg.norm(torch.matmul(A.t(), B)) ** 2

    normalization_x = torch.linalg.norm(torch.matmul(A.t(), A))
    normalization_y = torch.linalg.norm(torch.matmul(B.t(), B))

    cka = dot_product_similarity / (normalization_x * normalization_y)

    dot_product_similarity.detach()
    normalization_x.detach()
    normalization_y.detach()
    A.detach()
    B.detach()
    del dot_product_similarity, normalization_x, normalization_y, A, B
    return cka


def compute_variance(rep_stats: list, num_instances: int):
    """
    Computes the magnitude or concentricity variance for the given instance-wise statistics of activations and number of
    instances.

    :param rep_stats: The instance-wise length or density of each activation.
    :param num_instances: The total number of activations.
    :return: The calculated variance
    """
    rep_stats = torch.stack(rep_stats)
    max_sim, _ = torch.max(rep_stats, dim=0)
    min_sim, _ = torch.min(rep_stats, dim=0)
    mean_sim = torch.mean(rep_stats, dim=0)
    fct = 1 / (max_sim - min_sim)
    res = torch.sum(rep_stats - mean_sim, dim=0) / num_instances
    return fct * torch.nan_to_num(torch.sqrt(res))


def magnitude(act: torch.Tensor):
    """
    Computes the magnitude of a set of activations computed by a single model. The measure is defined as the length of
    the mean activation computed over the given instances.

    :param act: The tensor containing the model activations.
    :return: The computed magnitude.
    """
    mean_rep = torch.sum(act, dim=0) / len(act)
    mag = torch.linalg.norm(mean_rep)
    return mag


def magnitude_variance(activations: dict):
    """
    Computes the variance in magnitude of instance-wise activations computed by a set of models. Each entry in the
    dictionary is expected to contain a tensor of activations for a specific model.

    :param activations: The dictionary containing the model activations.
    :return: The computed magnitude variances.
    """
    euc_norm = []
    for entry in activations:
        euc_norm.append(torch.norm(torch.flatten(torch.vstack(activations[entry]), start_dim=1), p=2, dim=1))
    return compute_variance(euc_norm, len(activations))


def concentricity(act: torch.Tensor):
    """
    Computes the concentricity, a measure of the density of activations, by calculating the mean cosine similarity of
    all instances contained in the given tensor with the mean activation. The measure is calculated on the activations
    of a single model.

    :param act: The tensor containing the model activations.
    :return: The computed concentricity.
    """
    mean_rep = torch.sum(act, dim=0) / len(act)
    sims = cosine_similarity(act, mean_rep)
    return torch.sum(sims) / len(act)


def concentricity_variance(activations: dict):
    """
    Computes the variance in concentricity of instance-wise activations computed by a set of models. Each entry in the
    dictionary is expected to contain a tensor of activations for a specific model.

    :param activations: The dictionary containing the model activations.
    :return: The computed concentricity variances.
    """
    sims = []
    for entry in activations:
        act = torch.flatten(torch.vstack(activations[entry]), start_dim=1)
        mean_rep = torch.sum(act, dim=0) / len(activations[entry])
        sims.append(cosine_similarity(act, mean_rep))
    return compute_variance(sims, len(activations))


# Code taken from https://github.com/implicitDeclaration/similarity
def calculate_cosine_similarity_matrix(h_emb: torch.Tensor, eps: float = 1e-8):
    r'''
        h_emb: (N, M) hidden representations
    '''
    # normalize

    # h_emb = torch.tensor(h_emb)
    a_n = h_emb.norm(dim=1).unsqueeze(1)
    a_norm = h_emb / torch.max(a_n, eps * torch.ones_like(a_n))

    # cosine similarity matrix
    sim_matrix = torch.einsum('bc,cd->bd', a_norm, a_norm.transpose(0, 1))
    sim_matrix.fill_diagonal_(0)
    return sim_matrix


def second_order_cos_sim(acts1: torch.Tensor, acts2: torch.Tensor, center_columns: bool = True, eps: float = 1e-8):
    """
    Computes the second-order cosine similarity between two sets of activations of different models. For each set of
    activations, the pair-wise similarity between all instance within one set of activations are compared between the
    two sets and the mean value of all computed similarity scores is taken. Both times, cosine similarity is used to
    compute similarity. The measure is bounded in the interval [0, 1].

    :param acts1: The first set of activations.
    :param acts2: The second set of activations.
    :param center_columns: Boolean indicating whether columns should be mean-centered, default: True.
    :param eps: Small number to avoid zero-division in cosine similarity computation, default: 1e-8.
    :return: The computed second-order cosine similarity score.
    """
    assert len(acts1) == len(acts2)
    if center_columns:
        acts1 = acts1 - torch.mean(acts1, dim=0, keepdim=True)
        acts2 = acts2 - torch.mean(acts2, dim=0, keepdim=True)
    sim_matrix1 = calculate_cosine_similarity_matrix(acts1, eps)
    sim_matrix2 = calculate_cosine_similarity_matrix(acts2, eps)
    sims = cosine_similarity(sim_matrix1, sim_matrix2)
    res = torch.sum(sims) / len(acts1)
    del acts1, acts2, sim_matrix1, sim_matrix2, sims
    return res


# Code taken from https://github.com/js-d/sim_metric
def procrustes(A, B):
    """
    Computes Procrustes distance between representations A and B
    """
    A_sq_frob = torch.sum(A ** 2).cpu()
    B_sq_frob = torch.sum(B ** 2).cpu()

    mul = (A @ B.T).cpu().numpy()
    nuc = np.linalg.norm(mul, ord='nuc')
    res = A_sq_frob + B_sq_frob - 2 * torch.tensor(nuc)
    del A, B
    return res


def orthogonal_procrustes(acts1: torch.Tensor, acts2: torch.Tensor, center_columns: bool = True, use_norm: bool = True):
    """
    Computes the orthogonal procrustes similarity measure between two sets of activations of different models. The
    activations are scaled using Frobenius norm.

    :param acts1: The first set of activations.
    :param acts2: The second set of activations.
    :param center_columns: Boolean indicating whether columns should be mean-centered, default: True.
    :param use_norm: Boolean indicating whether the activation matrices should be normalized.
    :return: The computed similarity score.
    """
    assert len(acts1) == len(acts2)
    acts1 = torch.flatten(acts1, start_dim=1)
    acts2 = torch.flatten(acts2, start_dim=1)
    if center_columns:
        acts1 = acts1 - torch.mean(acts1, dim=0, keepdim=True)
        acts2 = acts2 - torch.mean(acts2, dim=0, keepdim=True)
    if use_norm:
        norm1 = torch.linalg.norm(acts1, ord='fro', dim=(0, 1))
        norm2 = torch.linalg.norm(acts2, ord='fro', dim=(0, 1))
        acts1 = acts1 / norm1
        acts2 = acts2 / norm2
    if acts1.shape[1] > acts2.shape[1]:
        acts_pad = pad(acts2, (0, acts1.shape[1] - acts2.shape[1]))
        return procrustes(acts1.T, acts_pad.T)
    elif acts1.shape[1] < acts2.shape[1]:
        acts_pad = pad(acts1, (0, acts2.shape[1] - acts1.shape[1]))
        return procrustes(acts_pad.T, acts2.T)
    return procrustes(acts1.T, acts2.T)


def compute_shift_shape(activations1: torch.Tensor, activations2: torch.Tensor, alpha: float = 1.0,
                        center_columns: bool = True):
    """
    Computes the shift shape similarity measure between two sets of activations of different models. They are expected
    to be four-dimensional.

    :param activations1: The first set of activations.
    :param activations2: The second set of activations.
    :param alpha: Regularization parameter, default: 1.0.
    :param center_columns: Boolean indicating whether a mean-centering operation should be learned, default: True.
    :return: The computed similarity score.
    """
    assert len(activations1) == len(activations2)
    assert len(activations1.shape) == len(activations2.shape) == 4

    # convert nchw to nhwc
    activations1 = torch.permute(activations1, (0, 2, 3, 1))
    activations2 = torch.permute(activations2, (0, 2, 3, 1))

    acts1, acts2 = ms.pad_activations(activations1, activations2)
    n, h, w, c = acts1.shape
    acts1 = acts1.reshape((n*h*w, c)).numpy()
    acts2 = acts2.reshape((n*h*w, c)).numpy()
    metric = LinearMetric(alpha=alpha, center_columns=center_columns)
    metric.fit(acts1, acts2)
    return metric.score(acts1, acts2)


def calculate_nearest_neighbors(acts1: torch.Tensor, acts2: torch.Tensor, center_columns: bool = True, k: int = 500,
                                sim_funct: Literal['euc', 'cos_sim'] = 'euc'):
    """
    Calculates the point-wise distances between each activation contained in the first tensor to each activation in the
    second tensor. Depending on the passed similarity function, euclidian distance or cosine similarity is used to
    calculate the distance. The indices of the k-nearest neighbors for each set of activations is then returned.

    :param acts1: The first set of activations.
    :param acts2: The second set of activations.
    :param center_columns: Boolean indicating whether columns should be mean-centered, default: True.
    :param k: The number of nearest neighbor indices to return for each activation, default: 500.
    :param sim_funct: The similarity function to be used to determine the nearest neighbors, default: 'euc'.
    :return: The indices of the k-nearest neighbors for each activation.
    """
    if sim_funct not in ['euc', 'cos_sim']:
        raise ValueError(f"Invalid similarity function: {sim_funct}. Please use 'euc' or 'cos_sim'.")

    assert len(acts1) == len(acts2)
    assert k <= len(acts1)
    acts1_mean, acts2_mean = acts1, acts2
    if center_columns:
        acts1_mean = acts1 - torch.mean(acts1, dim=0, keepdim=True)
        acts2_mean = acts2 - torch.mean(acts2, dim=0, keepdim=True)
    if sim_funct == 'euc':
        sim_matrix1 = torch.cdist(acts1_mean, acts1_mean, p=2)
        sim_matrix2 = torch.cdist(acts2_mean, acts2_mean, p=2)
    else:
        sim_matrix1 = calculate_cosine_similarity_matrix(acts1_mean)
        sim_matrix2 = calculate_cosine_similarity_matrix(acts2_mean)
    largest = sim_funct == 'cos_sim'
    indices1, indices2 = (torch.topk(sim_matrix1, k, largest=largest).indices,
                          torch.topk(sim_matrix2, k, largest=largest).indices)
    del acts1_mean, acts2_mean, sim_matrix1, sim_matrix2
    return indices1, indices2


# Code adapted from numpy.intersect1d() function
def get_rank_sum(indices1: np.ndarray, indices2: np.ndarray):
    """
    Computes the sum term for rank similarity given the two 1D-arrays containing the indices of the k-nearest neighbors
    of two sets of activations.

    :param indices1: One row of indices calculated for the first set of activations.
    :param indices2: One row of indices calculated for the second set of activations.
    :return: The calculated rank sum.
    """
    aux = np.concatenate((indices1, indices2))
    aux_sort_indices = np.argsort(aux, kind='mergesort')
    aux = aux[aux_sort_indices]
    mask = aux[1:] == aux[:-1]
    ar1_indices = aux_sort_indices[:-1][mask] + 1
    ar2_indices = aux_sort_indices[1:][mask] - indices1.size + 1
    rank_sum = np.sum([2 / ((1 + abs(i - j)) * (i + j)) for i, j in zip(ar1_indices, ar2_indices)])
    return rank_sum


def knn_jaccard(acts1: torch.Tensor, acts2: torch.Tensor, center_columns: bool = True, k: int = 500,
                sim_funct: Literal['euc', 'cos_sim'] = 'euc'):
    """
    Computes the k-NN Jaccard similarity between two sets of activations. For each set, the indices of the k-nearest
    neighbors for each activation within each set are determined and then used to calculate the intersection and union
    of the nearest neighbors for each pair of activations to compute the Jaccard similarity.

    :param acts1: The first set of activations.
    :param acts2: The second set of activations.
    :param center_columns: Boolean indicating whether columns should be mean-centered, default: True.
    :param k: The number of nearest neighbor indices to return for each activation, default: 500.
    :param sim_funct: The similarity function to be used to determine the nearest neighbors, default: 'euc'.
    :return: The pair-wise Jaccard similarity scores between the two sets.
    """
    assert len(acts1) == len(acts2)
    assert k <= len(acts1)
    indices1, indices2 = calculate_nearest_neighbors(acts1, acts2, center_columns, k, sim_funct)
    inds = torch.cat((indices1, indices2), dim=1)
    len_union = torch.Tensor([len(torch.unique(i)) for i in inds])
    len_intersection = torch.Tensor([len(set(i).intersection(set(j))) for i, j in zip(indices1.cpu().numpy(),
                                                                                      indices2.cpu().numpy())])
    res = len_intersection / len_union
    del indices1, indices2, inds, len_union, len_intersection
    return res


def rank_similarity(acts1: torch.Tensor, acts2: torch.Tensor, center_columns: bool = True, k: int = 500,
                    sim_funct: str = 'euc'):
    """
    Computes the rank similarity between two sets of activations. For each set, the indices of the k-nearest neighbors
    for each activation within each set are determined and then used to calculate the pair-wise rank rep_stats.

    :param acts1: The first set of activations.
    :param acts2: The second set of activations.
    :param center_columns: Boolean indicating whether columns should be mean-centered, default: True.
    :param k: The number of nearest neighbor indices to return for each activation, default: 500.
    :param sim_funct: The similarity function to be used to determine the nearest neighbors, default: 'euc'.
    :return: The pair-wise rank similarity scores between the two sets.
    """
    assert len(acts1) == len(acts2)
    assert k <= len(acts1)
    indices1, indices2 = calculate_nearest_neighbors(acts1, acts2, center_columns, k, sim_funct)
    rank_sums = [get_rank_sum(i, j) for i, j in zip(indices1.cpu().numpy(), indices2.cpu().numpy())]
    len_intersection = torch.Tensor([len(set(i).intersection(set(j))) for i, j in zip(indices1.cpu().numpy(),
                                                                                      indices2.cpu().numpy())])
    factors = []
    for idx, elem1 in enumerate(len_intersection):
        if elem1 > 0:
            factors.append(1 / sum([1 / (i + 1) for i in range(int(elem1))]))
        else:
            factors.append(0)
    res = torch.Tensor(factors) * torch.Tensor(rank_sums)
    del indices1, indices2, rank_sums, len_intersection, factors
    return res


def joint_rank_jaccard(acts1: torch.Tensor, acts2: torch.Tensor, center_columns: bool = True, k: int = 500,
                       sim_funct: str = 'euc'):
    """
    Computes the joint rank and k-NN Jaccard similarity between two sets of activations. For each set, the indices of
    the pairwise rank rep_stats and Jaccard rep_stats are calculated and multiplied to determine the final
    similarity scores.

    :param acts1: The first set of activations.
    :param acts2: The second set of activations.
    :param center_columns: Boolean indicating whether columns should be mean-centered, default: True.
    :param k: The number of nearest neighbor indices to return for each activation, default: 500.
    :param sim_funct: The similarity function to be used to determine the nearest neighbors, default: 'euc'.
    :return: The joint rank and Jaccard similarity score.
    """
    rank_sims = rank_similarity(acts1, acts2, center_columns, k, sim_funct)
    knn_sims = knn_jaccard(acts1, acts2, center_columns, k, sim_funct)
    res = torch.mean(knn_sims * rank_sims)
    del rank_sims, knn_sims
    return res


# Copied from https://github.com/sgstepaniants/GULP/blob/main/distance_functions.py
def _predictor_dist(A, B, evals_a=None, evecs_a=None, evals_b=None, evecs_b=None, lmbda=0):
    """
    Computes distance between best linear predictors on representations A and B
    """
    k, n = A.shape
    l, _ = B.shape
    assert k <= n
    assert l <= n

    if evals_a is None or evecs_a is None:
        evals_a, evecs_a = np.linalg.eigh(A @ A.T)
    if evals_b is None or evecs_b is None:
        evals_b, evecs_b = np.linalg.eigh(B @ B.T)

    evals_a = (evals_a + np.abs(evals_a)) / (2 * n)
    if lmbda > 0:
        inv_a_lmbda = np.array([1 / (x + lmbda) if x > 0 else 1 / lmbda for x in evals_a])
    else:
        inv_a_lmbda = np.array([1 / x if x > 0 else 0 for x in evals_a])

    evals_b = (evals_b + np.abs(evals_b)) / (2 * n)
    if lmbda > 0:
        inv_b_lmbda = np.array([1 / (x + lmbda) if x > 0 else 1 / lmbda for x in evals_b])
    else:
        inv_b_lmbda = np.array([1 / x if x > 0 else 0 for x in evals_b])

    T1 = np.sum(np.square(evals_a * inv_a_lmbda))
    T2 = np.sum(np.square(evals_b * inv_b_lmbda))

    cov_ab = A @ B.T / n
    T3 = np.trace(
        (np.diag(np.sqrt(inv_a_lmbda)) @ evecs_a.T)
        @ cov_ab
        @ (evecs_b @ np.diag(inv_b_lmbda) @ evecs_b.T)
        @ cov_ab.T
        @ (evecs_a @ np.diag(np.sqrt(inv_a_lmbda)))
    )

    return T1 + T2 - 2 * T3

# adapted from https://github.com/mklabunde/resi/blob/main/repsim/measures/gulp.py
def gulp(
    R: torch.Tensor,
    Rp: torch.Tensor,
    lmbda: float = 0,
) -> float:
    R, Rp = R.cpu().numpy(), Rp.cpu().numpy()

    # The GULP paper assumes DxN matrices; we have NxD matrices.
    n = R.shape[0]
    rep1 = R.T
    rep2 = Rp.T
    # They further assume certain normalization (taken from https://github.com/sgstepaniants/GULP/blob/d572663911cf8724ed112ee566ca956089bfe678/cifar_experiments/compute_dists.py#L82C5-L89C54)
    rep1 = rep1 - rep1.mean(axis=1, keepdims=True)
    rep1 = math.sqrt(n) * rep1 / np.linalg.norm(rep1)
    # center and normalize
    rep2 = rep2 - rep2.mean(axis=1, keepdims=True)
    rep2 = math.sqrt(n) * rep2 / np.linalg.norm(rep2)

    return _predictor_dist(rep1, rep2, lmbda=lmbda)
