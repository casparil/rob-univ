import argparse
import numpy as np
import pandas as pd
import torch
import os
import pickle

from univ.utils.measures import funct_sim
from univ.utils import similarity as sim
from univ.utils import load_data as ld
from univ.utils import measures as ms
from univ.utils import model_import as mi
from univ.utils import datasets as ds
from univ.utils import sampling as sa
from torchvision.datasets import CIFAR10
from robustness import data_augmentation as da
from loguru import logger


MEASURES = ['dis', 'norm', 'kappa', 'jsd', 'churn', 'amb', 'disc']
MEASURE_CHOICES = MEASURES + ['all']
MEASURE_NAMES = {
    'dis': 'dis',        # Disagreement
    'norm': 'norm_dis',  # Normalized disagreement
    'kappa': 'kappa',    # Cohen's kappa
    'jsd': 'jsd',        # Jensen-Shannon divergence
    'churn': 'churn',    # Surrogate churn
    'amb': 'amb',        # Ambiguity
    'disc': 'dis'        # Discrepancy
}
MEASURE_FOLDERS = {
    'dis': 'disagreement/',
    'norm': 'norm_dis/',
    'kappa': 'kappa/',
    'jsd': 'jsd/',
    'churn': 'churn/',
    'amb': 'ambiguity/',
    'disc': 'discrepancy/'
}
BASE_DIR = './'


parser = argparse.ArgumentParser(description='Calculate functional similarity on standard or inverted images.')
parser.add_argument('-a', '--alpha',
                    help='Alpha value for surrogate churn',
                    type=float,
                    default=1)
parser.add_argument('-b', '--batch',
                    help='Batch size',
                    type=int,
                    default=64)
parser.add_argument('-d', '--dataset',
                    help='The dataset to use',
                    choices=['sat6', 'imagenet', 'cifar10', 'snli', "imagenet100", "imagenet50"],
                    type=str,
                    default='imagenet')
parser.add_argument('-e', '--exp',
                    help='The experiment number',
                    type=int,
                    default=0)
parser.add_argument('-f', '--func',
                    help='Functional similarity measure to compute, defaults to all',
                    choices=MEASURE_CHOICES,
                    type=str,
                    default='all')
parser.add_argument('-i', '--inv',
                    help='Compute similarity on inverted images or textual adversarial examples',
                    choices=[0, 1],
                    type=int,
                    default=0)
parser.add_argument('-m', '--models',
                    help='The pre-trained models to compare',
                    choices=['imagenet', 'cifar10', 'snli', "imagenet100", "imagenet50"],
                    type=str,
                    default='imagenet')
parser.add_argument('-o', '--eps',
                    help='Epsilon for adversarial training',
                    type=str,
                    default='eps3')
parser.add_argument('-p', '--perturbs',
                    help='Number of times activations should be shuffled to calculate a baseline',
                    type=int,
                    default=0)
parser.add_argument('-s', '--sample',
                    help='Sample inverted images or adversarial examples',
                    choices=[0, 1],
                    type=int,
                    default=0)
parser.add_argument('-v', '--adv',
                    help='Compare models on ImageNet adversarial examples',
                    choices=[0, 1],
                    type=int,
                    default=0)
parser.add_argument("--model-dir",
                    default=None,
                    required=True,
                    help="Path to directory that contains all checkpoints of models of interest")
parser.add_argument("--index-file-dir", default=None)
parser.add_argument("--inverted-imgs-dir", required=True)
parser.add_argument("--config-file-path", default=None)
parser.add_argument("--labels-path", default="")
parser.add_argument("--data-path", default="")
parser.add_argument("--cache-dir", default="")
parser.add_argument("--no-comparisons", action="store_true", default=False)
parser.add_argument("--model-names", default=None, nargs="*")


def sample_labels(
        data: dict,
        all_indices: list,
        model_names: list,
        data_path: str,
        path_to_labels_file: str,
        path_to_index_file: str,
    ):
    if args.dataset in ["imagenet", "imagenet100", "imagenet50", "sat6"]:
        indices = list(pd.read_csv(path_to_index_file, index_col=0)['0'])
        labels = np.array(pd.read_csv(path_to_labels_file, header=None, names=['y'])).take(indices)
        if bool(args.sample):
            labels = ld.sample_labels(labels, all_indices, 10000, len(model_names) * 2)
        assert isinstance(labels, np.ndarray)
        labels = labels.tolist()
        print(len(labels))
    elif args.dataset == 'snli':
        labels = []
        for entry in data:
            labels.append(data[entry][2])
    elif args.dataset == 'cifar10':
        ds = CIFAR10(root=data_path, train=False, transform=da.TEST_TRANSFORMS_DEFAULT(32))
        labels = ds.targets
    else:
        raise NotImplementedError
    return labels


def get_predictions(models: list, dataloaders: list, device: torch.device):
    outputs = sim.get_outputs(models, dataloaders, device)
    preds = ms.get_class_predictions(outputs)
    probs = ms.get_probabilities(outputs)
    return preds, probs


def get_inverted_predictions(dataset_attr: dict, data: dict, model_names: list, models: list, device: torch.device,
                             indices: list, path_to_index_file: str):
    outputs = sim.get_outputs_inverted(dataset_attr['dict_paths'], model_names, models, data, device, args.dataset,
                                       args.batch)
    labels = sample_labels(data, indices, model_names, dataset_attr['data_path'], dataset_attr["labels_path"], path_to_index_file)
    preds = ms.get_predictions_inverted(outputs)
    probs = ms.get_probabilities_inverted(outputs)
    return preds, probs, labels


def get_outputs(
        models: list,
        device: torch.device,
        dataset_name: str,
        inverted_imgs_dir: str,
        dataset_attr: dict,
        epsilon: str,
        model_names: list,
        path_to_index_file: str,
        path_to_labels_file: str,
        batch_size: int,
    ):
    if bool(args.inv) or bool(args.adv):
        inverted_imgs = ld.load_inverted_data(model_names, inverted_imgs_dir, dataset_name, epsilon)
        inverted_imgs, indices = sa.sample_data(
            inverted_imgs,
            path_to_index_file,
            bool(args.adv),
            bool(args.sample),
            dataset_name
        )
        preds, probs, labels = get_inverted_predictions(
            dataset_attr,
            inverted_imgs,
            model_names,
            models,
            device,
            indices,
            path_to_index_file,
        )
    else:
        use_cifar = args.models == 'cifar10'
        data, labels, dataloader = ld.get_data(dataset_name, path_to_labels_file, dataset_attr['data_path'],
                                               path_to_index_file, dataset_attr['dict_paths'], batch_size, num_samples=10000, use_cifar=use_cifar)
        if args.dataset != 'cifar10':
            labels = np.asarray(labels)
            assert isinstance(labels, np.ndarray), f"labels should be numpy array, but is {type(labels)}"
            labels = labels.tolist()
        if args.dataset == 'imagenet':
            indices = list(pd.read_csv(path_to_index_file, index_col=0)['0'])
            labels = np.array(pd.read_csv(path_to_labels_file, header=None, names=['y'])).take(indices).tolist()

        preds, probs = get_predictions(models, dataloader, device)
    return preds, probs, labels


def get_stats(preds: dict, measure: str, model_names: list, path: str, eps: str, stats_type: str):
    if bool(args.inv) or bool(args.adv):
        stats = ms.get_inverted_statistics(preds, measure)
        ms.save_dataframe_to_csv(pd.DataFrame(np.array(stats)), model_names,
                                 f'{path}{MEASURE_NAMES[measure]}_{stats_type}{eps}.csv')
    else:
        if measure == 'amb':
            stats = funct_sim.ambiguity(preds)
        else:
            stats = funct_sim.discrepancy(preds)
        ms.save_dataframe_to_csv(pd.DataFrame([stats]), [measure],
                                 f'{path}{MEASURE_NAMES[measure]}_{stats_type}{eps}.csv')


def get_func_sim(outputs: dict, measure: str, model_names: list, path: str, eps: str, sim_type: str, labels: list,
                 num_classes: int):
    shape = (len(model_names), len(model_names))
    if bool(args.inv) or bool(args.adv):
        sims, sims_base = ms.get_inverted_funct_sim(outputs, measure, labels, num_classes, args.alpha,
                                                    args.perturbs > 0, args.perturbs)
    else:
        sims, sims_base = ms.get_funct_sim(outputs, measure, labels, num_classes, args.alpha, args.perturbs > 0,
                                           args.perturbs)
    df = pd.DataFrame(np.array(sims).reshape(shape), columns=model_names, index=model_names)
    ms.save_dataframe_to_csv(df, model_names, f'{path}{MEASURE_NAMES[measure]}_{sim_type}{eps}.csv')

    if sims_base:
        ms.save_baseline_scores(sims_base, f'{path}{MEASURE_NAMES[measure]}_{sim_type}all_{eps}.csv',
                                model_names, num_perturbations=args.perturbs)


def calculate_similarity(outputs: dict, measure: str, model_names: list, path: str, eps: str,
                         sim_type: str, labels: list, num_classes: int):
    print(f'Calculating {measure}')
    if measure in ['amb', 'disc']:
        get_stats(outputs, measure, model_names, path, f'{eps}_{args.exp}', sim_type)
    else:
        get_func_sim(outputs, measure, model_names, path, f'{eps}_{args.exp}', sim_type, labels, num_classes)


def main():
    sa.check_dataset_models(args.dataset, args.models)
    assert args.perturbs >= 0 and args.batch > 0
    dataset = args.dataset + '/'
    dataset_attr = ds.get_dataset_attr(dataset, args.models + '/', BASE_DIR, args.eps)
    args.adv = args.adv if args.dataset == 'imagenet' else 0

    model_names = args.model_names if args.model_names else dataset_attr['model_names']
    eps = dataset_attr['eps']
    exp_number = args.exp
    perturbs = args.perturbs
    sim_type = 'adv_' if bool(args.adv) else ''
    sim_type += 'sample_' if bool(args.sample) else ''
    sim_type += 'base_' if perturbs > 0 else ''
    num_classes = dataset_attr['num_classes']
    folder = 'inverted/' if bool(args.inv) or bool(args.adv) else 'standard/'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = mi.load_models(model_names, args.models + '/', args.model_dir, device)

    if args.index_file_dir:
        idx_path = os.path.join(args.index_file_dir, f"{dataset}indices_{exp_number}.csv")
    else:
        idx_path = f'{dataset_attr["results_dir"]}disagreement/standard/{dataset}indices_{exp_number}.csv'

    if args.labels_path:
        path_to_labels_file = args.labels_path
        dataset_attr["labels_path"] = args.labels_path
    else:
        path_to_labels_file = dataset_attr["labels_path"]

    if args.data_path:
        dataset_attr["data_path"] = args.data_path


    preds, probs, labels = get_outputs(
        models,
        device,
        args.dataset,
        args.inverted_imgs_dir,
        dataset_attr,
        eps,
        model_names,
        idx_path,
        path_to_labels_file,
        args.batch,
    )
    assert isinstance(labels, list)
    add_path = f'{args.models}/' if args.dataset == 'sat6' else ''

    if args.cache_dir:
        cache_path = os.path.join(args.cache_dir, f'output__{args.dataset}_{args.eps}_i{args.inv}.pkl')
        logger.info(f'Caching outputs to {cache_path}')
        with open(cache_path, 'wb') as f:
            pickle.dump({'preds': preds, 'probs': probs, 'labels': labels}, f)
        if args.no_comparisons:
            logger.info('Skipping comparisons')
            return

    if args.func == 'all':
        for measure in MEASURES:
            path = f'{dataset_attr["results_dir"]}{MEASURE_FOLDERS[measure]}{folder}{dataset}{add_path}'
            if measure in ['churn', 'jsd']:
                calculate_similarity(probs, measure, model_names, path, eps, sim_type, labels, num_classes)
            else:
                calculate_similarity(preds, measure, model_names, path, eps, sim_type, labels, num_classes)
    else:
        path = f'{dataset_attr["results_dir"]}{MEASURE_FOLDERS[args.func]}{folder}{dataset}{add_path}'
        if args.func in ['churn', 'jsd']:
            calculate_similarity(probs, args.func, model_names, path, eps, sim_type, labels, num_classes)
        else:
            calculate_similarity(preds, args.func, model_names, path, eps, sim_type, labels, num_classes)


if __name__ == '__main__':
    args = parser.parse_args()
    main()
