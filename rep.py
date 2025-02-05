import argparse
import numpy as np
import pandas as pd
import torch
import os.path
import pickle
from loguru import logger

from univ.utils import similarity as sim
from univ.utils import load_data as ld
from univ.utils import measures as ms
from univ.utils import model_import as mi
from univ.utils import datasets as ds
from univ.utils import sampling as sa
from univ.utils.attacker import AttackerModel

from typing import List, Dict, Any

MEASURES = ['mag', 'con', 'cos', 'jacc', 'rank', 'rank_jacc', 'proc', 'cka', "gulp", "rtd"]
# MEASURES = ['mag', 'con', 'cos', 'jacc', 'rank', 'rank_jacc', 'proc', 'shape', 'cka', "gulp", "rtd"]
MEASURE_CHOICES = MEASURES + ['all'] + ["paper"]
MEASURE_NAMES = {
    'mag': 'mag',          # Magnitude
    'con': 'con',          # Concentricity
    'cos': 'cos_sim',      # 2nd order cosine similarity
    'jacc': 'jac',         # Jaccard similarity
    'rank': 'rank',        # Rank similarity
    'rank_jacc': 'joint',  # Joint k-NN Jaccard and rank similarity
    'proc': 'proc',        # Orthogonal Procrustes
    'shape': 'shape',      # Shape metric
    'cka': 'cka',           # Centered kernel alignment
    "gulp": "gulp",
    "rtd": "rtd",
}
MEASURE_FOLDERS = {
    'mag': 'magnitude/',
    'con': 'concentricity/',
    'cos': 'cos_sim/',
    'jacc': 'jaccard/',
    'rank': 'rank/',
    'rank_jacc': 'joint/',
    'proc': 'procrustes/',
    'shape': 'shape/',
    'cka': 'cka/',
    "gulp": "gulp/",
    "rtd": "rtd/",
}
BASE_DIR = './'


parser = argparse.ArgumentParser(description='Calculate representational similarity on standard or inverted images.')
parser.add_argument('-a', '--alpha',
                    help='Alpha value for the shift shape metric',
                    type=float,
                    default=1)
parser.add_argument('-b', '--batch',
                    help='Batch size',
                    type=int,
                    default=64)
parser.add_argument('-c', '--center',
                    help='Mean-center representations before calculating similarity',
                    choices=[0, 1],
                    type=int,
                    default=1)
parser.add_argument('-d', '--dataset',
                    help='The dataset to use',
                    choices=['sat6', 'imagenet', 'cifar10', 'snli', "imagenet100", "imagenet50"],
                    type=str,
                    default='imagenet')
parser.add_argument('-e', '--exp',
                    help='The experiment number',
                    type=float,
                    default=0)
parser.add_argument('-f', '--function',
                    help='Similarity function to use for nearest neighbor calculation',
                    choices=['euc', 'cos_sim'],
                    type=str,
                    default='cos_sim')
parser.add_argument('-i', '--inv',
                    help='Compute similarity on inverted images or textual adversarial examples',
                    choices=[0, 1],
                    type=int,
                    default=0)
parser.add_argument('-k', '--knn',
                    help='The number of nearest neighbors to consider',
                    type=int,
                    default=500)
parser.add_argument('--gulp-lambda',
                    help='Lambda value for the GULP measure',
                    type=float,
                    default=0)
parser.add_argument('-m', '--models',
                    help='The pre-trained models to compare',
                    choices=['imagenet', 'cifar10', 'snli', "imagenet100", "imagenet50"],
                    type=str,
                    default='imagenet')
parser.add_argument('-n', '--norm',
                    help='Normalize activations for Procrustes',
                    choices=[0, 1],
                    type=int,
                    default=1)
parser.add_argument('-o', '--eps',
                    help='Epsilon for adversarial training',
                    type=str,
                    default='eps3')
parser.add_argument('-p', '--perturbs',
                    help='Number of times activations should be shuffled to calculate a baseline',
                    type=int,
                    default=0)
parser.add_argument('-r', '--rep',
                    help='Representational similarity measure to compute, defaults to all',
                    choices=MEASURE_CHOICES,
                    type=str,
                    default='all')
parser.add_argument('-s', '--sample',
                    help='Sample inverted images or adversarial examples',
                    action="store_true",
                    default=False)
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
parser.add_argument("--num-inputs", default=10000, type=int)
parser.add_argument("--cache-dir", default="")
parser.add_argument("--no-comparisons", action="store_true", default=False)

def get_activations(
        models: List[AttackerModel],
        layers: List[str],
        device: torch.device,
        dataset_name: str,
        inverted_imgs_dir: str,
        dataset_attr: dict,
        epsilon: str,
        num_inputs: int,
        batch_size: int,
        model_names: List[str],
        use_inverted_imgs_or_adv_examples: bool,
        use_imagenet_adv_examples: bool,
        path_to_index_file: str,
        path_to_labels_file: str,
        path_to_input_data: str,
        pool_images_across_models: bool,
    ):
    """
    path_to_input_data (str): A path to an lmdb file or a directory of images.
    """
    if use_inverted_imgs_or_adv_examples or use_imagenet_adv_examples:
        if not epsilon.startswith("eps"):
            raise ValueError(f"epsilon should start with 'eps' and look like, e.g., 'eps025', but is {epsilon}")
        inverted_imgs = ld.load_inverted_data(model_names, inverted_imgs_dir, dataset_name, epsilon)
        inverted_imgs, _ = sa.sample_data(
            inverted_imgs,
            path_to_index_file,
            use_imagenet_adv_examples,
            pool_images_across_models,
            dataset_name,
        )
        activations = sim.inverted_activations(
            dataset_attr['dict_paths'],  # only relevant for SNLI models
            model_names,
            models,
            layers,
            inverted_imgs,
            device,
            dataset_name,
            batch_size=batch_size,
        )
    else:
        use_cifar = args.models == 'cifar10'
        _, _, dataloader = ld.get_data(
            dataset_name,
            path_to_labels_file,
            path_to_input_data,
            path_to_index_file,
            dataset_attr['dict_paths'],  # irrelevant for image models
            batch_size=batch_size,
            use_cifar=use_cifar,
            num_samples=num_inputs,
        )
        activations = sim.get_activations(models, layers, dataloader, device)
    return activations


def get_statistics(activations: dict, measure: str, model_names: list, path: str, eps: str, stats_type: str = ''):
    if bool(args.inv) or bool(args.adv):
        stats = ms.get_inverted_rep_sim_statistics(activations, measure)
        df = pd.DataFrame(np.array(stats).reshape((len(model_names), len(model_names))), columns=model_names,
                          index=model_names)
    else:
        stats = ms.get_rep_sim_statistics(activations, measure)
        df = pd.DataFrame(np.array(stats), index=model_names)
    ms.save_dataframe_to_csv(df, model_names, f'{path}{measure}_{stats_type}{eps}.csv')


def get_rep_sim(
    activations: dict,
    measure: str,
    model_names: list,
    path: str,
    eps: str,
    device: torch.device,
    sim_type: str,
    normalize_acts_procrustes: bool,
    center_acts: bool,
    baseline_shuffles: int,
    use_inverted_imgs_or_adv_examples: bool,
    use_imagenet_adv_examples: bool,
    knn: int,
    nn_simfunc: str,
    gulp_lambda: float,
    overwrite: bool = False,
):
    shape = (len(model_names), len(model_names))
    pre = ''
    if measure == 'proc':
        pre += 'norm_' if normalize_acts_procrustes else ''
    if measure != 'cka':
        pre += 'mean_' if center_acts else ''
    csv_path = f'{path}{MEASURE_NAMES[measure]}_{pre}{sim_type}{eps}.csv'
    if os.path.exists(csv_path) and not overwrite:
        logger.info(f"Results already exist at {csv_path} and overwrite=False. Skipping computation.")
        return
    if use_inverted_imgs_or_adv_examples or use_imagenet_adv_examples:
        if measure == 'cka':
            sims, sims_base = sim.row_cka(activations, device, baseline_shuffles > 0, baseline_shuffles)
        else:
            sims, sims_base = ms.get_inverted_rep_sim(
                activations,
                device,
                center_columns=center_acts,
                k=knn,
                sim_funct=nn_simfunc,
                measure=measure,
                permute=baseline_shuffles > 0,
                n_permutations=baseline_shuffles,
                use_norm=normalize_acts_procrustes,
                gulp_lambda=gulp_lambda,
            )
    else:
        if measure == 'cka':
            sims, sims_base = sim.pairwise_cka(activations, device, baseline_shuffles > 0, baseline_shuffles)
        else:
            sims, sims_base = ms.get_rep_sim(activations, device, center_columns=center_acts, k=knn,
                                             sim_funct=nn_simfunc, measure=measure, permute=baseline_shuffles > 0,
                                             n_permutations=baseline_shuffles, use_norm=normalize_acts_procrustes,
                                             gulp_lambda=gulp_lambda)
    sims = np.array([x for x in sims if x is not None])
    sims_base = [x for x in sims_base if x is not None]
    df = pd.DataFrame(sims.reshape(shape), columns=model_names, index=model_names)
    ms.save_dataframe_to_csv(df, model_names, csv_path)
    if len(sims_base) > 0:
        ms.save_baseline_scores(sims_base, f'{path}{MEASURE_NAMES[measure]}_{pre}{sim_type}all_{eps}.csv',
                                model_names, num_perturbations=baseline_shuffles)


def calculate_similarity(
        acts: Dict[int, Any],
        measure: str,
        model_names: list,
        path: str,
        eps: str,
        device: torch.device,
        sim_type: str,
        normalize_acts_procrustes: bool,
        center_acts: bool,
        baseline_shuffles: int,
        use_inverted_imgs_or_adv_examples: bool,
        use_imagenet_adv_examples: bool,
        knn: int,
        nn_simfunc: str,
        gulp_lambda: float,
    ):
    if measure in ['mag', 'con']:
        print(f'Calculating {measure}')
        get_statistics(acts, measure, model_names, path, f'{eps}_{args.exp}', sim_type)
    else:
        get_rep_sim(
            acts,
            measure,
            model_names,
            path,
            f'{eps}_{args.exp}',
            device,
            sim_type,
            normalize_acts_procrustes,
            center_acts,
            baseline_shuffles,
            use_inverted_imgs_or_adv_examples,
            use_imagenet_adv_examples,
            knn,
            nn_simfunc,
            gulp_lambda=gulp_lambda,
        )


def get_save_path(results_dir: str, measure: str):
    folder = 'inverted/' if bool(args.inv) or bool(args.adv) else 'standard/'
    add_path = f'{args.models}/' if args.dataset == 'sat6' else ''

    if measure in ['jacc', 'rank', 'rank_jacc']:
        path = f'{results_dir}{MEASURE_FOLDERS[measure]}{folder}{args.function}/{args.dataset}/{add_path}'
    elif measure == 'cka':
        path = f'{results_dir}{MEASURE_FOLDERS[measure]}{folder}10000/{args.dataset}/{add_path}'
    else:
        path = f'{results_dir}{MEASURE_FOLDERS[measure]}{folder}{args.dataset}/{add_path}'

    return path


def main():
    sa.check_dataset_models(args.dataset, args.models)
    assert args.perturbs >= 0 and args.batch > 0
    assert 0 < args.knn < 10000
    dataset = args.dataset + '/'
    dataset_attr = ds.get_dataset_attr(dataset, args.models + '/', BASE_DIR, args.eps)
    args.adv = args.adv if args.dataset == 'imagenet' else False

    model_names = dataset_attr['model_names']
    layers = dataset_attr['layers']
    four_dim_models = dataset_attr['four_dim_models']
    eps = dataset_attr['eps']
    exp_number = args.exp
    perturbs = args.perturbs
    sim_type = 'adv_' if bool(args.adv) else ''
    sim_type += 'sample_' if bool(args.sample) else ''
    sim_type += 'base_' if perturbs > 0 else ''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = mi.load_models(model_names, args.models + '/', args.model_dir, device)

    if args.index_file_dir:
        idx_path = os.path.join(args.index_file_dir, f"{dataset}indices_{exp_number}.csv")
    else:
        idx_path = f'{dataset_attr["results_dir"]}magnitude/standard/{dataset}indices_{exp_number}.csv'

    if args.labels_path:
        dataset_attr["labels_path"] = args.labels_path

    if args.data_path:
        dataset_attr["data_path"] = args.data_path

    activations = get_activations(
        models=models,
        model_names=model_names,
        layers=layers,
        device=device,
        dataset_name=args.dataset,
        dataset_attr=dataset_attr,
        epsilon=args.eps,
        inverted_imgs_dir=args.inverted_imgs_dir,
        path_to_index_file=idx_path,
        use_inverted_imgs_or_adv_examples=args.inv,
        use_imagenet_adv_examples=args.adv,
        pool_images_across_models=args.sample,
        batch_size=args.batch,
        path_to_labels_file=dataset_attr['labels_path'],
        path_to_input_data=dataset_attr['data_path'],
        num_inputs=args.num_inputs,
        )

    # avg pool activations for densenet
    densenet_idx = model_names.index("densenet161") if "densenet161" in model_names else None
    if densenet_idx is not None:
        if args.inv or args.adv:
            for inverted_idx, activations_per_inverted_dataset in activations.items():
                activations_per_inverted_dataset[densenet_idx] = [
                    batch_acts.mean(dim=(-1,-2)) for batch_acts in activations_per_inverted_dataset[densenet_idx]
                ]
        else:
            activations[densenet_idx] = [
                batch_acts.mean(dim=(-1,-2)) for batch_acts in activations[densenet_idx]
            ]

    logger.debug(f"{model_names=}")
    logger.debug(f"{four_dim_models=}")

    if args.cache_dir:
        cache_path = os.path.join(args.cache_dir, f'reps__{args.dataset}_{args.eps}_i{args.inv}.pkl')
        logger.info(f'Caching outputs to {cache_path}')
        with open(cache_path, 'wb') as f:
            pickle.dump({'activations': activations}, f)
        if args.no_comparisons:
            logger.info('Skipping comparisons')
            return

    if args.rep == 'all':
        for measure in MEASURES:
            path = get_save_path(dataset_attr['results_dir'], measure)
            if not os.path.exists(path):
                logger.info(f"Creating directory for results: {path}")
                os.makedirs(path)
            models = four_dim_models if measure == 'shape' else model_names
            calculate_similarity(
                activations,
                measure,
                models,
                path,
                eps,
                device,
                sim_type,
                args.norm,
                args.center,
                args.perturbs,
                bool(args.inv),
                bool(args.adv),
                args.knn,
                args.function,
                gulp_lambda=args.gulp_lambda,
            )
    elif args.rep=="paper":
        for measure in ["jacc", "proc", "cka", "rtd"]:
            path = get_save_path(dataset_attr['results_dir'], measure)
            if not os.path.exists(path):
                logger.info(f"Creating directory for results: {path}")
                os.makedirs(path)
            models = four_dim_models if measure == 'shape' else model_names
            calculate_similarity(
                activations,
                measure,
                models,
                path,
                eps,
                device,
                sim_type,
                args.norm,
                args.center,
                args.perturbs,
                bool(args.inv),
                bool(args.adv),
                args.knn,
                args.function,
                gulp_lambda=args.gulp_lambda,
            )
    else:
        path = get_save_path(dataset_attr['results_dir'], args.rep)
        if not os.path.exists(path):
            logger.info(f"Creating directory for results: {path}")
            os.makedirs(path)
        models = four_dim_models if args.rep == 'shape' else model_names
        calculate_similarity(
            activations,
            args.rep,
            models,
            path,
            eps,
            device,
            sim_type,
            args.norm,
            args.center,
            args.perturbs,
            bool(args.inv),
            bool(args.adv),
            args.knn,
            args.function,
            gulp_lambda=args.gulp_lambda,
        )

    if args.config_file_path is not None:
        import json
        if os.path.exists(args.config_file_path):
            logger.info(f"Config already exists under {args.config_file_path}. Only printing output:")
            logger.info(args.__dict__)
        else:
            with open(args.config_file_path, "w") as f:
                json.dump(args.__dict__, f)

if __name__ == '__main__':
    args = parser.parse_args()
    main()
