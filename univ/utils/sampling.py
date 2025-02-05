import random

from univ.utils import load_data as ld

from typing import Dict, Any, Tuple, List


def check_dataset_models(dataset: str, models: str):
    """
    Asserts that the given dataset string matches with the models. Dataset and model strings have to match except for
    ImageNet, where the dataset can be either SAT-6 or ImageNet.

    :param dataset: The name of the dataset to be used.
    :param models: The dataset on which the pre-trained models to be loaded were trained.
    """
    if dataset == 'imagenet':
        assert models == 'imagenet'
    elif dataset == 'imagenet100' or dataset == "imagenet50":
        pass
        # assert models == 'imagenet100'
    elif dataset == 'cifar10':
        assert models == 'cifar10'
    elif dataset == 'sat6':
        assert models in ['imagenet', 'cifar10']
    elif dataset == 'snli':
        assert models == 'snli'
    else:
        raise NotImplementedError(f'Combination of models {models} and dataset {dataset} not supported!')


def sample_data(
        data: Dict[int, Any], path_to_index_file: str, data_contains_adv_examples: bool, pool_images: bool, dataset: str
    ) -> Tuple[Dict[int, Any], List[int]]:
    """
    Samples 10000 entries form the given dictionaries of robust and standard data if models should be compared on
    adversarial examples. If all models are to be compared on the same sets of inverted images or adversarial examples,
    a fixed number of data points summing up to about 10000 is taken from each entry in the robust and standard data
    dictionaries and returned. If no sampling is required, the dictionaries are returned as is.

    :param data_standard: The data points on which standard models are to be compared.
    :param data_robust: The data points on which robust models are to be compared.
    :param path: The path where the indices of the used data points should be saved or the path to a file containing
    indices to be loaded.
    :param adv: Boolean indicating whether the data dictionaries contain adversarial examples.
    :param sample: Boolean indicating whether the data points should be sampled from all dictionary entries.
    :param dataset: The dataset for which data should be sampled.
    :return: The sampled data.
    """
    all_indices = []
    sample_size = 10_000

    if data_contains_adv_examples:
        for idx in data:
            indices = random.sample(range(len(data[idx])), sample_size)
            data[idx] = data[idx][indices]

    if pool_images:
        data, all_indices = ld.subsample_inverted_data(data, path_to_index_file, sample_size, sample_size, dataset)

    return data, all_indices
