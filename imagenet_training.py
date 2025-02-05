import argparse
from typing import Any
import torch
import timm
import numpy as np
import datetime
import os.path
import pandas as pd

from loguru import logger
from torch.utils.data import DataLoader
from univ.rob import datasets
from robustness import data_augmentation as da
from univ.utils import training
from univ.utils import ImageFolderLMDB
from univ.utils import model_import as mi

parser = argparse.ArgumentParser(description='Train a model on ImageNet1k or ImageNet100.')
parser.add_argument('-a', '--adv',
                    help='Adversarial (1) or standard training (0)',
                    choices=[0, 1],
                    type=int,
                    default=0)
parser.add_argument('-b', '--batch',
                    help='Batch size',
                    type=int,
                    default=256)
parser.add_argument('-c', '--constraint',
                    help='Constraing for adversarial training',
                    choices=['inf', '2', 'unconstrained', 'fourier'],
                    type=str,
                    default='2')
parser.add_argument('-d', '--data',
                    help='Path to ImageNet train and validation files',
                    type=str,
                    default='./data/imagenet/ILSVRC/')
parser.add_argument('-e', '--epochs',
                    help='Number of epochs to train',
                    type=int,
                    default=90)
parser.add_argument('-k', '--ckpt',
                    help='Save additional checkpoints every k epochs',
                    type=int,
                    default=5)
parser.add_argument('-l', '--lr',
                    help='Adversarial attack step size',
                    type=str,
                    default='2')
parser.add_argument('-m', '--model',
                    help='The pre-trained model to load',
                    choices=['resnet18', 'resnet50', 'wide_resnet50_2', 'wide_resnet50_4', 'densenet161',
                             'resnext50_32x4d', 'vgg16_bn', 'tiny_vit_5m'],
                    type=str,
                    default='tiny_vit_5m')
parser.add_argument("--model-lr", default=0.1, type=float)
parser.add_argument('-n', '--num',
                    help='The number of classes',
                    choices=[50, 100, 1000],
                    type=int,
                    default=1000)
parser.add_argument('-o', '--eps',
                    help='Epsilon for adversarial training',
                    type=float,
                    default=3)
parser.add_argument('-p', '--steps',
                    help='Number of attack steps for adversarial training',
                    type=int,
                    default=3)
parser.add_argument('-r', '--steplr',
                    help='Drop learning rate by 10 every r epochs',
                    type=int,
                    default=30)
parser.add_argument('-s', '--save',
                    help='Path to save model checkpoints',
                    type=str,
                    default='./data/cnns/imagenet/')
parser.add_argument('-t', '--train',
                    help='Train model (1) or evaluate (0)',
                    choices=[0, 1],
                    type=int,
                    default=1)
parser.add_argument('-v', '--eval',
                    help='Path to stored model checkpoint',
                    type=str,
                    default=None)
parser.add_argument('-w', '--workers',
                    help='Number of workers',
                    type=int,
                    default=4)
parser.add_argument("--results-file", type=str, default=None)


def train_model(model: str, resume_path: str, train_loader: DataLoader, val_loader: DataLoader, device: torch.device,
                dataset: datasets.DataSet, kwargs: dict):
    vit = True if model == 'tiny_vit_5m' else False
    if vit:
        model = timm.create_model('tiny_vit_5m_224.in1k', pretrained=False)
    att_model, checkpoint = mi.make_and_restore_model(arch=model, dataset=dataset, device=device, vit=vit,
                                                      resume_path=resume_path)
    current_epoch = checkpoint['epoch'] if checkpoint else 0
    training.train_model(kwargs, att_model, (train_loader, val_loader), devices=[device], current_epoch=current_epoch)


def eval_model(arch: str, path: str, val_loader: DataLoader, device: torch.device, dataset: datasets.DataSet,
               kwargs: dict) -> dict:
    assert path is not None
    vit = True if arch == 'tiny_vit_5m' else False
    if vit:
        arch = timm.create_model('tiny_vit_5m_224.in1k', pretrained=False)
    model, _ = mi.make_and_restore_model(arch=arch, dataset=dataset, device=device, resume_path=path, vit=vit)
    return training.eval_model(kwargs, model, val_loader, None, [device])

def convert_eps_str_to_float(s: str) -> float:
    if s == "eps0":
        return 0.0
    elif s == "eps025":
        return 0.25
    elif s == "eps05":
        return 0.5
    elif s == "eps1":
        return 1.0
    elif s == "eps3":
        return 3.0
    else:
        raise ValueError(f"Unknown eps identifier: {s}")

def main():
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = training.get_kwargs(args)

    if args.num == 1000:
        dataset = datasets.ImageNet('')
    elif args.num == 100:
        dataset = datasets.ImageNet100('')
    elif args.num == 50:
        dataset = datasets.ImageNet50('')
    else:
        raise ValueError(f"Num classes must be one of 1000,100,50")

    val = ImageFolderLMDB(args.data + 'val.lmdb', da.TEST_TRANSFORMS_IMAGENET)
    val_loader = DataLoader(val, batch_size=args.batch, shuffle=True, num_workers=args.workers)

    if bool(args.train):
        train = ImageFolderLMDB(args.data + 'train.lmdb', da.TRAIN_TRANSFORMS_IMAGENET)
        train_loader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=args.workers)
        train_model(args.model, args.eval, train_loader, val_loader, device, dataset, kwargs)
    else:
        results = eval_model(args.model, args.eval, val_loader, device, dataset, kwargs)

        if args.results_file:
            trained_eps = convert_eps_str_to_float(os.path.split(os.path.split(args.eval)[0])[1])
            # Save result to file
            new_data = {
                "model": [args.model] * 2,
                "dataset": [dataset.ds_name] * 2,
                "acc": [results["nat_prec1"], results["adv_prec1"]],
                "loss": [results["nat_loss"], results["adv_loss"]],
                "trained_eps": [trained_eps] * 2,
                "attack_lr": [0, args.lr],
                "attack_eps": [0, args.eps],
                "timestamp": [str(datetime.datetime.now())] * 2
            }
            new_data = pd.DataFrame.from_dict(new_data)

            if args.results_file is None:
                logger.info("Specify --results-file to save evaluation results to file")
            elif os.path.exists(args.results_file):
                logger.info("Found previous evaluation data. Updating...")
                old_data = pd.read_csv(args.results_file, index_col=0)
                pd.concat((old_data, new_data), axis="index").to_csv(args.results_file)
            else:
                logger.info(f"Found no previous evaluation data. Creating new file at {args.results_file}")
                # os.mkdir(os.path.dirname(args.results_file),)
                new_data.to_csv(args.results_file)


if __name__ == '__main__':
    args = parser.parse_args()
    main()
