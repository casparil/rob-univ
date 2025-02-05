import dill
import os
import os.path
import timm
import torch

from univ.utils.attacker import AttackerModel
from univ.rob import datasets
from univ.rift.models import AdvPLM
from univ.rift import utils


# Code take from https://github.com/MadryLab/robustness/blob/master/robustness/model_utils.py
class DummyModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, *args, **kwargs):
        return self.model(x)


# Adapted import method for pre-trained models as some of the files seem to contain some errors.
def make_and_restore_model(*_, arch, dataset, resume_path=None, parallel=False, pytorch_pretrained=False,
                           add_custom_forward=False, device=None, to_cifar10=False, vit=False):
    """
    Makes a model and (optionally) restores it from a checkpoint.
    Args:
        arch (str|nn.Module): Model architecture identifier or otherwise a
            torch.nn.Module instance with the classifier
        dataset (Dataset class [see datasets.py])
        resume_path (str): optional path to checkpoint saved with the
            robustness library (ignored if ``arch`` is not a string)
        not a string
        parallel (bool): if True, wrap the model in a DataParallel
            (defaults to False)
        pytorch_pretrained (bool): if True, try to load a standard-trained
            checkpoint from the torchvision library (throw error if failed)
        add_custom_forward (bool): ignored unless arch is an instance of
            nn.Module (and not a string). Normally, architectures should have a
            forward() function which accepts arguments ``with_latent``,
            ``fake_relu``, and ``no_relu`` to allow for adversarial manipulation
            (see `here`<https://robustness.readthedocs.io/en/latest/example_usage/training_lib_part_2.html#training-with-custom-architectures>
            for more info). If this argument is True, then these options will
            not be passed to forward(). (Useful if you just want to train a
            model and don't care about these arguments, and are passing in an
            arch that you don't want to edit forward() for, e.g.  a pretrained model)
    Returns:
        A tuple consisting of the model (possibly loaded with checkpoint), and the checkpoint itself
    """
    if (not isinstance(arch, str)) and add_custom_forward:
        arch = DummyModel(arch)

    classifier_model = dataset.get_model(arch, pytorch_pretrained) if \
        isinstance(arch, str) else arch

    model = AttackerModel(classifier_model, dataset, device, vit=vit)

    if to_cifar10:
        assert arch in ['wide_resnet50_2', 'wide_resnet50_4', 'resnext50_32x4d']
        num_ftrs = model.model.fc.in_features
        model.model.fc = torch.nn.Linear(num_ftrs, 10)

    # optionally resume from a checkpoint
    checkpoint = None
    if resume_path and os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path, pickle_module=dill, map_location=device)

        # Makes us able to load models saved with legacy versions
        state_dict_path = 'model'
        if not ('model' in checkpoint):
            state_dict_path = 'state_dict'

        sd = checkpoint[state_dict_path]
        sd = {k[len('module.'):]: v for k, v in sd.items()}
        sd_updated = {}
        # loop through all items to replace all items named 'model.model' with 'model' for the load function to succeed
        for name, value in sd.items():
            if 'model.model.' in name:
                updated_name = name.replace('model.model.', 'model.')
                sd_updated[updated_name] = value
            else:
                sd_updated[name] = value
        model.load_state_dict(sd_updated)
        print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
    elif resume_path:
        error_msg = "=> no checkpoint found at '{}'".format(resume_path)
        raise ValueError(error_msg)

    if parallel:
        model = torch.nn.DataParallel(model, device_ids=device)
    if device is not None:
        model = model.to(device)
    else:
        model = model.cuda()

    return model, checkpoint


def import_cifar_model(arch: str, device: torch.device, resume_path: str = None):
    """
    Loads an untrained model architecture for training on CIFAR-10 or a model trained on CIFAR-10, if a path to a
    checkpoint is given. As some model architectures are not supported by the CIFAR-10 dataset class, the corresponding
    models are loaded using the ImageNet dataset and then modifying the last model layer to output 10 instead of 1000
    predictions.

    :param arch: The architecture of the model, supported are 'resnet18', 'resnet50', 'wide_resnet50_2',
    'wide_resnet50_4', 'resnext50_32x4d', 'densenet161' and 'vgg16'.
    :param device: The device to which the model should be loaded.
    :param resume_path: An optional to a checkpoint saved with the robustness library.
    :return: The loaded model.
    """
    assert arch in ['resnet18', 'resnet50', 'wide_resnet50_2', 'wide_resnet50_4', 'resnext50_32x4d', 'densenet161',
                    'vgg16']
    if arch in ['wide_resnet50_2', 'wide_resnet50_4', 'resnext50_32x4d']:
        model, _ = make_and_restore_model(arch=arch, dataset=datasets.ImageNet(''), device=device,
                                          resume_path=resume_path, to_cifar10=True)
    else:
        model, _ = make_and_restore_model(arch=arch, dataset=datasets.CIFAR(''), resume_path=resume_path,
                                          device=device)
    return model


def import_snli_model(arch: str, device: torch.device, resume_path: str):
    """
    Imports a BERT, RoBERTa or DistilBERT model expected to be fine-tuned on SNLI.

    :param arch: The architecture of the model, supported are 'bert', 'roberta', and 'distilbert'.
    :param device: The device to which the model should be loaded.
    :param resume_path: The path to a model checkpoint.
    :return: The loaded model.
    """
    assert arch in ['bert', 'roberta', 'distilbert']
    opt = {
        'plm_type': arch,
        'mixout_p': 0,
        'device_ids': [device],
        'freeze_plm': True,
        'freeze_plm_teacher': True,
        'infonce_sim_metric': 'normal',
        'label_size': 3,
        'infonce_temperature': 0.2,
    }
    model = AdvPLM(opt)
    utils.set_params(model, resume_path, device=device)
    return model


def load_models(names: list, dataset: str, model_dir: str, device: torch.device) -> list:
    """
    Loads models with the specified names from the given paths where they are saved for the given
    dataset.

    :param names: The names of the models to load, i.e. their file names without filetype extension.
    :param dataset: The dataset on which the models were trained, supported are ImageNet, CIFAR-10 and SNLI.
    :param model_dir: The path to the folder where the models are saved.
    :param device: The device to which the models should be loaded.
    :return: The loaded robust and non-robust models.
    """
    models = []

    dataset = dataset if dataset.endswith("/") else dataset + "/"

    for name in names:
        if dataset == 'imagenet/':
            if name == 'tiny_vit_5m':
                arch = timm.create_model('tiny_vit_5m_224.in1k', pretrained=False)
                vit = True
            else:
                arch = name
                vit = False
            model, _ = make_and_restore_model(
                arch=arch,
                dataset=datasets.ImageNet(''),
                device=device,
                resume_path=os.path.join(model_dir, f"{name}.ckpt"),
                vit=vit,
            )
        elif dataset == "imagenet100/":
            if name == 'tiny_vit_5m':
                arch = timm.create_model('tiny_vit_5m_224.in1k', pretrained=False)
                vit = True
            else:
                arch = name
                vit = False
            model, _ = make_and_restore_model(
                arch=arch,
                dataset=datasets.ImageNet100(''),
                device=device,
                resume_path=os.path.join(model_dir, f"{name}.ckpt"),
                vit=vit,
            )
        elif dataset == "imagenet50/":
            if name == 'tiny_vit_5m':
                arch = timm.create_model('tiny_vit_5m_224.in1k', pretrained=False)
                vit = True
            else:
                arch = name
                vit = False
            model, _ = make_and_restore_model(
                arch=arch,
                dataset=datasets.ImageNet50(''),
                device=device,
                resume_path=os.path.join(model_dir, f"{name}.ckpt"),
                vit=vit,
            )
        elif dataset == 'cifar10/':
            model = import_cifar_model(arch=name, device=device, resume_path=model_dir + name + '.pt')
        elif dataset == 'snli/':
            model = import_snli_model(arch=name, device=device, resume_path=model_dir + name + '.pth')
        else:
            raise NotImplementedError
        models.append(model)
    return models