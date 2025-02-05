import argparse
import datetime
import os
import pandas as pd
import torch

from loguru import logger
from univ.utils import model_import as mi
from univ.utils import training
from univ.rob import datasets

parser = argparse.ArgumentParser(description='Train a CIFAR-10 model.')
parser.add_argument('-a', '--adv',
                    help='Adversarial (1) or standard training (0)',
                    choices=[0, 1],
                    type=int,
                    default=0)
parser.add_argument('-b', '--batch',
                    help='Batch size',
                    type=int,
                    default=512)
parser.add_argument('-c', '--constraint',
                    help='Constraing for adversarial training',
                    choices=['inf', '2', 'unconstrained', 'fourier'],
                    type=str,
                    default='2')
parser.add_argument('-d', '--data',
                    help='Path to CIFAR-10 data',
                    type=str,
                    default='./data/cifar10/')
parser.add_argument('-e', '--epochs',
                    help='Number of epochs to train',
                    type=int,
                    default=90)
parser.add_argument('-k', '--ckpt',
                    help='Save additional checkpoints every k epochs',
                    type=int,
                    default=15)
parser.add_argument('-l', '--lr',
                    help='Adversarial attack step size',
                    type=str,
                    default='0.5')
parser.add_argument('-m', '--model',
                    help='CNN architecture to train',
                    choices=['resnet18', 'resnet50', 'wide_resnet50_2', 'wide_resnet50_4', 'densenet161',
                             'resnext50_32x4d', 'vgg16'],
                    type=str,
                    default='resnet18')
parser.add_argument("--model-lr", default=0.1, type=float)
parser.add_argument('-o', '--eps',
                    help='Epsilon for adversarial training',
                    type=float,
                    default=1)
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
                    default='./data/cnns/cifar10/')
parser.add_argument('-t', '--train',
                    help='Train model (1) or evaluate (0)',
                    choices=[0, 1],
                    type=int,
                    default=1)
parser.add_argument('-v', '--eval',
                    help='Path to stored model checkpoint',
                    type=str,
                    default='./data/cnns/cifar10/eps1/resnet18.pt')
parser.add_argument('-w', '--workers',
                    help='Number of workers',
                    type=int,
                    default=4)
parser.add_argument("--results-file", type=str, default=None)


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
    kwargs = training.get_kwargs(args, 'cifar', args.model)

    data_path = os.path.expandvars(kwargs.data)
    dataset = datasets.DATASETS[kwargs.dataset](data_path)
    train_loader, val_loader = dataset.make_loaders(kwargs.workers, kwargs.batch_size, data_aug=bool(kwargs.data_aug))

    if bool(args.train):
        model = mi.import_cifar_model(kwargs.arch, device)
        training.train_model(kwargs, model, (train_loader, val_loader), devices=[device])
    else:
        model = mi.import_cifar_model(kwargs.arch, device, args.eval)
        results = training.eval_model(kwargs, model, val_loader, None, [device])

        if args.results_file:
            trained_eps = convert_eps_str_to_float(os.path.split(os.path.split(args.eval)[0])[1])
            # Save result to file
            new_data = {
                "model": [args.model] * 2,
                "dataset": ["cifar10"] * 2,
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
