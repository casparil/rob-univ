import argparse
import pandas as pd
import random
import timm
import torch
import os.path
from typing import Optional
from loguru import logger

from univ.utils import model_import as mi
from univ.utils import load_data as ld
from univ.utils import similarity as sim
from univ.utils import datasets as ds
from univ.utils import sampling as sa
from torchvision import transforms
from torchvision.datasets import CIFAR10
from univ.rob.datasets import ImageNet, ImageNet100, ImageNet50
from robustness import data_augmentation as da
from loguru import logger


BASE_DIR = './'


parser = argparse.ArgumentParser(description='Generate inverted images.')
parser.add_argument('-b', '--batch',
                    help='Batch size',
                    type=int,
                    default=64)
parser.add_argument('-d', '--dataset',
                    help='The dataset to use',
                    choices=['sat6', 'imagenet', 'cifar10', 'imagenet100', "imagenet50"],
                    type=str,
                    default='imagenet')
parser.add_argument('-e', '--eps',
                    help='The robustness level of the model to be loaded',
                    choices=['eps0', 'eps1', 'eps3', 'eps05', 'eps025'],
                    type=str,
                    default='eps0')
parser.add_argument('-i', '--inverted-imgs-path',
                    help='Path to save inverted images',
                    type=str,
                    default='./data/imagenet/inverted/')
parser.add_argument('-m', '--model',
                    help='The pre-trained model to load',
                    choices=['resnet18', 'resnet50', 'wide_resnet50_2', 'wide_resnet50_4', 'densenet161',
                             'resnext50_32x4d', 'vgg16', 'vgg16_bn', 'tiny_vit_5m'],
                    type=str,  # 'vgg16' (CIFAR-10), 'vgg16_bn' (ImageNet), 'tiny_vit_5m' (ImageNet)
                    default='resnet18')
parser.add_argument('-n', '--num-imgs-to-invert',
                    help='The number of indices to generate',
                    type=int,
                    default=0)
parser.add_argument('-p', '--pretrain-dataset',
                    help='The dataset on which the model was pre-trained',
                    choices=['imagenet', 'cifar10', 'imagenet100', "imagenet50"],
                    type=str,
                    default='imagenet')
parser.add_argument('-s', '--seed-indices-path',
                    help='Path to seed indices file',
                    type=str,
                    default='./results/cka/inverted/10000/imagenet/seed_indices_0.csv')
parser.add_argument('-t', '--target-indices-path',
                    help='Path to target indices file',
                    type=str,
                    default='./results/cka/inverted/10000/imagenet/target_indices_0.csv')
parser.add_argument('-u', '--upper-bound-idx',
                    help='The upper bound for indices if seed and target indices should be generated',
                    type=int,
                    default=0)  # max. 9999 for CIFAR-10, 79999 for SAT-6, 49999 for ImageNet, 129999 for ImageNet100, 64999 for ImageNet50
parser.add_argument("--uint8", action="store_true", default=False,
                     help="If specified, the inverted images are stored in uint8 format to save on storage")
parser.add_argument("--model-dir", default=None)
parser.add_argument("--data-path", default=None)
parser.add_argument("--labels-path", default=None)
parser.add_argument("--random-state", type=int, default=573098435098230)
parser.add_argument(
    "--overwrite",
    help="Overwrite inverted image file if it already exists. By default stop if it exists.",
    action="store_true",
    default=False
)



def get_target_indices(max_idx: int, num_to_sample: int, target_indices_path: str):
    if max_idx > 0 and num_to_sample > 0:
        logger.info(
            f"Sampling {num_to_sample} images as target images for inversion. "
            f"Their index is in the range [0, {max_idx}]. "
            f"Result will be stored as csv at {target_indices_path}."
        )
        indices = random.sample(range(max_idx), num_to_sample)
        ld.save_indices(indices, target_indices_path)
    else:
        indices = list(pd.read_csv(target_indices_path, index_col=0)['0'])
    return indices


def get_seed_indices(max_idx: int, num_to_sample: int, labels: list, labels_path: Optional[str], dataset: str, seed_indices_path: str, random_state: int):
    if max_idx > 0 and num_to_sample > 0:
        indices = sim.get_seed_indices(labels, labels_path, upper_bound=max_idx, dataset=dataset, random_state=random_state)
        ld.save_indices(indices, seed_indices_path)
    else:
        indices = list(pd.read_csv(seed_indices_path, index_col=0)['0'])
    return indices


def get_target_images(model_path: str, labels_path: str, images_path: str, indices: list, device: torch.device):
    if args.pretrain_dataset == 'imagenet' or args.pretrain_dataset == 'imagenet100' or args.pretrain_dataset == 'imagenet50':
        transform = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()])
    else:
        transform = da.TEST_TRANSFORMS_DEFAULT(32)

    if args.pretrain_dataset == 'imagenet':
        # Import pre-trained ImageNet model and get target images
        if args.model == 'tiny_vit_5m':
            arch = timm.create_model('tiny_vit_5m_224.in1k', pretrained=False)
            model, _ = mi.make_and_restore_model(arch=arch, dataset=ImageNet(''),
                                                 resume_path=model_path + args.model + '.ckpt', device=device, vit=True)
        else:
            model, _ = mi.make_and_restore_model(arch=args.model, dataset=ImageNet(''),
                                                 resume_path=model_path + args.model + '.ckpt', device=device)
        if args.dataset == 'sat6':
            images, labels = ld.load_sat6_images(labels_path, images_path, transform, indices)
        else:
            images, labels = ld.load_images(labels_path, images_path, transform, 10, indices)
        return model, torch.stack(images), labels, None
    elif args.pretrain_dataset == 'imagenet100':
        # Import pre-trained ImageNet100 model and get target images
        if args.model == 'tiny_vit_5m':
            arch = timm.create_model('tiny_vit_5m_224.in1k', pretrained=False)
            model, _ = mi.make_and_restore_model(arch=arch, dataset=ImageNet100(''),
                                                 resume_path=model_path + args.model + '.ckpt', device=device, vit=True)
        else:
            model, _ = mi.make_and_restore_model(arch=args.model, dataset=ImageNet100(''),
                                                 resume_path=model_path + args.model + '.ckpt', device=device)
        images, labels = ld.load_imagenet100_or_50_images(images_path, transform, indices)
        return model, torch.stack(images), labels, None
    elif args.pretrain_dataset == 'imagenet50':
        # Import pre-trained ImageNet50 model and get target images
        if args.model == 'tiny_vit_5m':
            arch = timm.create_model('tiny_vit_5m_224.in1k', pretrained=False)
            model, _ = mi.make_and_restore_model(arch=arch, dataset=ImageNet50(''),
                                                 resume_path=model_path + args.model + '.ckpt', device=device, vit=True)
        else:
            model, _ = mi.make_and_restore_model(arch=args.model, dataset=ImageNet50(''),
                                                 resume_path=model_path + args.model + '.ckpt', device=device)
        images, labels = ld.load_imagenet100_or_50_images(images_path, transform, indices)
        return model, torch.stack(images), labels, None
    else:
        # Import pre-trained CIFAR-10 model and get target images
        model = mi.import_cifar_model(args.model, device, model_path + args.model + '.pt')
        if args.dataset == 'sat6':
            images, labels = ld.load_sat6_images(labels_path, images_path, transform, indices)
            return model, torch.stack(images), labels, None
        else:
            cif_data = CIFAR10(root=images_path, train=False, transform=transform)
            labels = cif_data.targets
            images = ld.transform_cifar_images(cif_data.data, transform)
            return model, torch.stack(images), labels, cif_data


def get_seed_images(labels_path: str, images_path: str, indices: list, cif_data: CIFAR10):
    if args.pretrain_dataset == 'imagenet' or args.pretrain_dataset == "imagenet100" or args.pretrain_dataset == 'imagenet50':
        transform = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()])
    else:
        transform = da.TEST_TRANSFORMS_DEFAULT(32)

    if args.dataset == 'imagenet':
        seed_img, seed_labels = ld.load_images(labels_path, images_path, transform, 10, indices)
    elif args.dataset == 'imagenet100' or args.pretrain_dataset == 'imagenet50':
        seed_img, seed_labels = ld.load_imagenet100_or_50_images(images_path, transform, indices)
    elif args.dataset == 'cifar10':
        seed_img, seed_labels = ([cif_data.data[i] for i in indices], [cif_data.targets[i] for i in indices])
        seed_img = ld.transform_cifar_images(list(seed_img), da.TEST_TRANSFORMS_DEFAULT(32))
    elif args.dataset == 'sat6':
        seed_img, seed_labels = ld.load_sat6_images(labels_path, images_path, transform, indices)
    else:
        raise NotImplementedError

    return torch.stack(seed_img), seed_labels


def main():
    print(args)
    save_path_inv_imgs = f'{args.inverted_imgs_path}{args.model}_{args.eps}.pt'
    if os.path.exists(save_path_inv_imgs) and not args.overwrite:
        logger.info(f"Inverted images already exist at {save_path_inv_imgs}. Add '--overwrite' to overwrite.")
        return

    random.seed(args.random_state)
    logger.info(f"Seed set to {args.random_state}")
    assert args.upper_bound_idx >= 0 and args.num_imgs_to_invert >= 0
    sa.check_dataset_models(args.dataset, args.pretrain_dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_attr = ds.get_dataset_attr(args.dataset + '/', args.pretrain_dataset + '/', BASE_DIR, args.eps)

    robust_path = dataset_attr['robust_path']
    standard_path = dataset_attr['standard_path']
    model_dir = standard_path if args.eps == 'eps0' else robust_path
    images_path = dataset_attr['data_path']  # soll auf train.lmdb für in100 zeigen
    labels_path = dataset_attr['labels_path']  # irrelevant für in100. labels sind in lmdb file


    if args.model_dir is not None:
        model_dir = args.model_dir
        logger.info(f"Loading model from {model_dir}")
    if args.data_path is not None:
        images_path = args.data_path
        logger.info(f"Loading images from {images_path}")
    if args.labels_path is not None:
        labels_path = args.labels_path
        logger.info(f"Loading labels from {labels_path}")

    target_indices = get_target_indices(args.upper_bound_idx, args.num_imgs_to_invert, args.target_indices_path)
    model, target_images, labels, data = get_target_images(model_dir, labels_path, images_path, target_indices, device)  # labels_path is irrelevant for in100

    path = None if args.dataset == 'cifar10' else labels_path
    seed_indices = get_seed_indices(
        args.upper_bound_idx, args.num_imgs_to_invert, labels, path, args.dataset, args.seed_indices_path, random.randint(0, int(1e10)))
    seed_images, seed_labels = get_seed_images(labels_path, images_path, seed_indices, data)

    inverted = sim.get_inversions(model, seed_images, target_images, args.batch, device=device)
    sim.save_inverted(save_path_inv_imgs, inverted, args.uint8)


if __name__ == '__main__':
    args = parser.parse_args()
    main()
