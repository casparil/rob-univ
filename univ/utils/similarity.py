import gc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import torch
import tqdm
from loguru import logger
from typing import Union, Optional

from univ.image_inversion import invert_images
from random import shuffle
from univ.rift.models import AdvPLM
from robustness.attacker import AttackerModel
from univ.measures.rep_sim import cka
from univ.utils.minibatch_cka import MinibatchCKA
from univ.utils import load_data as ld
from torch.utils.data import DataLoader


def get_activation(activations: list, check_appended: bool = False):
    """
    Creates a hook that appends the activation output of the respective model to the given list.

    :param activations: The list of activations.
    :param check_appended: Boolean indicating whether the last appended tensor should be checked for equalitiy to
    current outputs, necessary for TinyViT as function is called two times, default: False.
    :return: The hook.
    """
    def hook(model, input, output):
        if not check_appended or len(activations) == 0:
            activations.append(output.detach().to('cpu'))
        elif not torch.equal(activations[-1], output.cpu()):
            activations.append(output.detach().to('cpu'))
    return hook


def register_hook(model: any, layer_name: str, activations: list):
    """
    Registers a forward hook returning the output for the given model at the layer with the given name.

    :param model: The model for which the hook should be registered.
    :param layer_name: The name of the layer at which the hook should be registered.
    :param activations: The list in which the output should be stored.
    :return: The registered hook.
    """
    model.eval()
    for name, layer in model.model.named_modules():
        if name == layer_name:
            return layer.register_forward_hook(get_activation(activations, layer_name == 'head.drop'))


def register_hook_snli(model: AdvPLM, layer_name: str, activations: list):
    """
    Registers a forward hook returning the output for the given model at the layer with the given name.

    :param model: The model for which the hook should be registered.
    :param layer_name: The name of the layer at which the hook should be registered.
    :param activations: The list in which the output should be stored.
    :return: The registered hook.
    """
    model.plm.eval()
    model.cls_to_logit.eval()
    for name, layer in model.named_modules():
        if name == layer_name:
            return layer.register_forward_hook(get_activation(activations))


def get_image_activations(models: list, dataloader: DataLoader, device: torch.device):
    """
    Computes the activations at the penultimate layer of the given models for the images passed in the dataloader. The
    activations are stored in a dictionary containing lists of batch-wise activations for each model.

    :param models: The list of models for which the activations should be returned.
    :param dataloader: The dataloader providing the image batches.
    :param device: The device on which the forward pass should be computed, preferably a GPU.
    """
    for image_batch, _ in tqdm.tqdm(dataloader, desc='| Computing activations |', total=len(dataloader)):
        image_batch = image_batch.to(device)
        for model in models:
            model(image_batch)
        image_batch.detach()
        del image_batch


def get_snli_activations(models: list, dataloaders: list, device: torch.device):
    """
    Computes the activations at the penultimate layer of the given models for the SNLI data passed in the dataloader.
    The activations are stored in a dictionary containing lists of batch-wise activations for each model.

    :param models: The list of models for which the activations should be returned.
    :param dataloaders: A list containing one dataloader per model providing the SNLI data.
    :param device: The device on which the forward pass should be computed, preferably a GPU.
    """
    for idx, loader in enumerate(dataloaders):
        for snli_batch in tqdm.tqdm(loader, desc='| Computing activations |', total=len(loader)):
            text_x = snli_batch[0].to(device)
            attention_mask = snli_batch[4].to(device)
            models[idx](text_x, attention_mask)
            text_x.detach()
            attention_mask.detach()
            del text_x, attention_mask


def get_activations(models: list, layers: list, dataloaders: list, device: torch.device):
    """
    Returns the activations at the penultimate layer of the given models for the data passed in the dataloader. Every
    model is set to evaluation mode and registered with a forward hook for returning the desired output which is stored
    in a dictionary containing lists of batch-wise activations for each model.

    :param models: The list of models for which the activations should be returned.
    :param layers: The name of one layer for each of the models to monitor.
    :param dataloaders: A list containing a single dataloader for CNNs and one dataloader per model for LLMs.
    :param device: The device on which the forward pass should be computed, preferably a GPU.
    :return: The computed activation dictionary.
    """
    activations = {index: [] for index in range(len(models))}
    if len(dataloaders) == 1:
        for idx, model in enumerate(models):
            register_hook(model, layers[idx], activations[idx])
            models[idx] = model.to(device)
        get_image_activations(models, dataloaders[0], device)
    else:
        assert len(dataloaders) == len(models)
        for idx, model in enumerate(models):
            register_hook_snli(model, layers[idx], activations[idx])
            models[idx] = model.to(device)
        get_snli_activations(models, dataloaders, device)
    torch.cuda.empty_cache()
    return activations


def inverted_image_activations(models: list, layers: list, inverted_images: dict, device: torch.device,
                               batch_size: int = 64):
    """
    Returns the activations of the layers specified in the given layers list for the given models. The activations are
    computed using the inverted images provided as a dictionary, where each entry corresponds to images generated by a
    specific model. For each entry, the activations across all models are computed and stored in a dictionary.

    :param models: The list of models for which the activations should be returned.
    :param layers: The name of one layer for each of the models to monitor.
    :param inverted_images: The dictionary containing the inverted images as torch tensors.
    :param device: The device on which the forward pass should be computed.
    :param batch_size: The batch size used to split the image tensors, default: 64.
    :return: A dictionary containing all activations for all models on each entry in the image dictionary.
    """
    all_activations = {}
    for entry in inverted_images:
        activations = {index: [] for index in range(len(models))}
        hooks = []
        for idx, model in enumerate(models):
            hooks.append(register_hook(model, layers[idx], activations[idx]))
            models[idx] = model.to(device)
        inverted_batches = inverted_images[entry].split(batch_size)
        for batch in tqdm.tqdm(inverted_batches, desc='| Activations for model {} |'.format(entry),
                               total=len(inverted_batches)):
            inverted = batch.to(device)
            for idx, model in enumerate(models):
                model(inverted)
            inverted.detach()
            del inverted
        for hook in hooks:
            hook.remove()
        for idx, model in enumerate(models):
            models[idx] = model.to('cpu')
        torch.cuda.empty_cache()
        all_activations[entry] = activations.copy()
    return all_activations


def inverted_snli_activations(dict_paths: list, model_names: list, models: list, layers: list, data: dict,
                              device: torch.device, batch_size: int = 64):
    """
    Returns the activations of the layers specified in the given layers list for the given models. The activations are
    computed using the data provided as a dictionary, containing the premises, hypotheses and labels and a
    model-specific dataloader. The resulting activations are store in a dictionary and returned.

    :param dict_paths: The paths to the dictionaries for all models.
    :param model_names: The names of the models.
    :param models: The list of models for which the activations should be returned.
    :param layers: The name of one layer for each of the models to monitor.
    :param data: The dictionary containing the SNLI data.
    :param device: The device on which the forward pass should be computed.
    :param batch_size: The batch size, default: 64.
    :return: A dictionary containing all activations for all models.
    """
    all_activations = {}
    for entry, model in enumerate(models):
        loaders = []
        hooks = []
        activations = {index: [] for index in range(len(models))}
        for idx, model in enumerate(models):
            hooks.append(register_hook_snli(model, layers[idx], activations[idx]))
            models[idx] = model.to(device)
            model.plm.eval()
            model.cls_to_logit.eval()
        for idx, path in enumerate(dict_paths):
            loaders.append(ld.get_snli_data(path, data[entry][0], data[entry][1], data[entry][2], batch_size,
                                            model_names[idx]))
        get_snli_activations(models, loaders, device)
        for hook in hooks:
            hook.remove()
        all_activations[entry] = activations.copy()
    return all_activations


def inverted_activations(dict_paths: list, model_names: list, models: list, layers: list, data: dict,
                         device: torch.device, dataset: str, batch_size: int = 64):
    """
    Returns the activations of the layers specified in the given layers list for the given models. The activations are
    computed using the data provided as a dictionary and returned.

    :param dict_paths: The paths to the dictionaries for all models for SNLI.
    :param model_names: The names of the models.
    :param models: The list of models for which the activations should be returned.
    :param layers: The name of one layer for each of the models to monitor.
    :param data: The dictionary containing the data.
    :param device: The device on which the forward pass should be computed.
    :param dataset: The dataset for which the outputs should be computed.
    :param batch_size: The batch size to use, default: 64.
    :return: A dictionary containing all activations for all models on each entry in the image dictionary.
    """
    if dataset == 'snli':
        assert len(dict_paths) == len(models)
        return inverted_snli_activations(dict_paths, model_names, models, layers, data, device, batch_size)
    else:
        return inverted_image_activations(models, layers, data, device, batch_size)


def get_image_outputs(models: list, dataloader: DataLoader, device: torch.device):
    """
    Returns the outputs of the given models for the image data passed in the dataloader. A dictionary containing the
    predictions of all models is returned.

    :param models: The list of models for which the outputs should be returned.
    :param dataloader: The dataloader for fetching data.
    :param device: The device on which the forward pass should be computed.
    :return: The computed output dictionary.
    """
    outputs = {index: [] for index in range(len(models))}
    for image_batch, _ in tqdm.tqdm(dataloader, desc='| Computing outputs |', total=len(dataloader)):
        image_batch = image_batch.to(device)
        for idx, model in enumerate(models):
            outputs[idx].append(model(image_batch, with_image=False).detach().cpu())
        image_batch.detach()
        del image_batch
    return outputs


def get_snli_outputs(models: list, dataloaders: list, device: torch.device):
    """
    Returns the outputs of the given models for the SNLI data passed in the dataloader. A dictionary containing the
    predictions of all models is returned.

    :param models: The list of models for which the outputs should be returned.
    :param dataloaders: A dataloader for each model to fetch data.
    :param device: The device on which the forward pass should be computed.
    :return: The computed output dictionary.
    """
    outputs = {index: [] for index in range(len(models))}
    for idx, loader in enumerate(dataloaders):
        for snli_batch in tqdm.tqdm(loader, desc='| Computing outputs |', total=len(loader)):
            text_x = snli_batch[0].to(device)
            attention_mask = snli_batch[4].to(device)
            outputs[idx].append(models[idx](text_x, attention_mask).detach().cpu())
            text_x.detach()
            attention_mask.detach()
            del text_x, attention_mask
    return outputs


def get_outputs(models: list, dataloaders: list, device: torch.device):
    """
    Returns the outputs of the given models for the data passed in the dataloaders. Every model is set to evaluation
    mode and the desired output stored in a dictionary containing the stacked outputs of each model is returned.

    :param models: The list of models for which the outputs should be returned.
    :param dataloaders: A list of dataloaders for fetching data.
    :param device: The device on which the forward pass should be computed.
    :return: The computed output dictionary.
    """
    for idx, model in enumerate(models):
        models[idx] = model.to(device)
        model.eval()
    if len(dataloaders) == 1:
        outputs = get_image_outputs(models, dataloaders[0], device)
    else:
        assert len(dataloaders) == len(models)
        for idx, model in enumerate(models):
            model.plm.eval()
            model.cls_to_logit.eval()
        outputs = get_snli_outputs(models, dataloaders, device)
    for idx, model in enumerate(models):
        outputs[idx] = torch.vstack(outputs[idx])
    torch.cuda.empty_cache()
    return outputs


def get_inverted_image_outputs(models: list, inverted_images: dict, device: torch.device, batch_size: int = 64):
    """
    Returns the outputs for the given models using the inverted images provided as a dictionary, where each entry
    corresponds to images generated by a specific model. For each entry, the outputs across all models are computed and
    stored in a dictionary.

    :param models: The list of models for which the outputs should be returned.
    :param inverted_images: The dictionary containing the inverted images as torch tensors.
    :param device: The device on which the forward pass should be computed.
    :param batch_size: The batch size used to split the image tensors, default: 64.
    :return: A dictionary containing all outputs for all models on each entry in the image dictionary.
    """
    all_outputs = {}
    for entry in inverted_images:
        outputs = {index: [] for index in range(len(models))}
        for idx, model in enumerate(models):
            models[idx] = model.to(device)
            model.eval()
        inverted_batches = inverted_images[entry].split(batch_size)
        for batch in tqdm.tqdm(inverted_batches, desc='| Outputs for model {} |'.format(entry),
                               total=len(inverted_batches)):
            inverted = batch.to(device)
            for idx, model in enumerate(models):
                outputs[idx].append(model(inverted, with_image=False).detach().cpu())
            inverted.detach()
            del inverted
        for idx, model in enumerate(models):
            outputs[idx] = torch.vstack(outputs[idx])
        gc.collect()
        torch.cuda.empty_cache()
        all_outputs[entry] = outputs
    return all_outputs


def get_inverted_snli_outputs(dict_paths: list, model_names: list, models: list, data: dict, device: torch.device,
                              batch_size: int = 64):
    """
    Returns the outputs for the given models using the SNLI data provided as a dictionary.

    :param dict_paths: The paths to the dictionaries for all models.
    :param model_names: The names of the models.
    :param models: The list of models for which the outputs should be returned.
    :param data: The dictionary containing the data.
    :param device: The device on which the forward pass should be computed.
    :param batch_size: The batch size, default: 64.
    :return: A dictionary containing all outputs for all models.
    """
    all_outputs = {}
    for idx, model in enumerate(models):
        models[idx] = model.to(device)
        model.plm.eval()
        model.cls_to_logit.eval()
    for entry in data:
        loaders = []
        for idx, path in enumerate(dict_paths):
            loaders.append(ld.get_snli_data(path, data[entry][0], data[entry][1], data[entry][2], batch_size,
                                            model_names[idx]))
        outputs = get_snli_outputs(models, loaders, device)
        for idx, _ in enumerate(model_names):
            outputs[idx] = torch.vstack(outputs[idx])
        all_outputs[entry] = outputs
    return all_outputs


def get_outputs_inverted(dict_paths: list, model_names: list, models: list, data: dict, device: torch.device,
                         dataset: str, batch_size: int = 64):
    """
    Returns outputs for data for which spurious correlations have been removed and returns the output on this data for
    each of the given models.

    :param dict_paths: The paths to the dictionaries for all models.
    :param model_names: The names of the models.
    :param models: The list of models for which the outputs should be returned.
    :param data: The dictionary containing the data.
    :param device: The device on which the forward pass should be computed.
    :param dataset: The dataset for which the outputs should be computed.
    :param batch_size: The batch size, default: 64.
    :return: A dictionary containing all outputs for all models.
    """
    if dataset == 'snli':
        return get_inverted_snli_outputs(dict_paths, model_names, models, data, device, batch_size)
    else:
        return get_inverted_image_outputs(models, data, device, batch_size)


def compute_permuted_cka(activations1: list, activations2: list, device: torch.device, n_permutations: int = 10):
    """
    Randomly permutes the given second model activations and computes the CKA score between the permuted values and the
    other activations. This process is repeated according to the given number of permutations and the average score is
    returned.

    :param activations1: The activations of the first model.
    :param activations2: The activations of the second model.
    :param device: The device to which the activations should be transferred.
    :param n_permutations: The number of random permutations to compute for an average result, default: 10.
    :return: The average of the computed scores.
    """
    scores = []
    for i in range(n_permutations):
        permuted = activations2.copy()
        shuffle(permuted)
        # scores.append(compute_cka(activations1, permuted, device))
        scores.append(compute_minibatch_cka(activations1, permuted))
    return np.mean(scores), scores


def compute_minibatch_cka(activations1: list, activations2: list):
    """
    Computes CKA similarity for two lists of the same size containing batches of activations.

    :param activations1: The list of activations of the first model.
    :param activations2: The list of activations of the second model.
    :return: The computed result.
    """
    assert len(activations1) == len(activations2)
    minibatch_cka = MinibatchCKA(1, 1, across_models=True)
    minibatch_cka.update_state_across_models([torch.vstack(activations1)], [torch.vstack(activations2)])
    res = minibatch_cka.result().numpy()[0][0]
    del minibatch_cka
    return res


def compute_cka(activations1: list, activations2: list, device: torch.device):
    """
    Computes CKA similarity for two lists of the same size containing batches of activations.

    :param activations1: The list of activations of the first model.
    :param activations2: The list of activations of the second model.
    :param device: The device to which the activations should be transferred.
    :return: The computed result.
    """
    act1 = torch.flatten(torch.vstack(activations1), start_dim=1).to(device)
    act2 = torch.flatten(torch.vstack(activations2), start_dim=1).to(device)
    result = cka(act1, act2)
    act1.detach()
    act2.detach()
    del act1, act2
    gc.collect()
    torch.cuda.empty_cache()
    return result.to('cpu')


def pairwise_cka(activations: dict, device: torch.device, permute: bool = False, n_permutations: int = 10):
    """
    Computes the pairwise CKA values for the given dictionary of activations where every entry corresponds to a list of
    batch-wise activations of one model. If the same activations entries are compared, a value of one is returned. If
    the CKA value for two activation lists has already been computed, the value is simply extracted from the list
    current CKA values list and appended again. Otherwise, the CKA value is computed on the two activations lists.

    :param activations: The activation dictionary containing an entry of activations for each model.
    :param device: The device to which the activations should be transferred.
    :param permute: Whether to compute a baseline score on permuted inputs, default: False.
    :param n_permutations: The number of random permutations to compute for an average result, default: 10.
    :return: The list of computed CKA values.
    """
    ckas, base = [], []
    for entry1 in activations:
        for entry2 in tqdm.tqdm(activations, desc='| CKA for model {} |'.format(entry1), total=len(activations)):
            if permute:
                mean_scores, scores = compute_permuted_cka(activations[entry1], activations[entry2], device,
                                                           n_permutations)
                ckas.append(mean_scores)
                base.append(scores)
            else:
                if entry1 == entry2:
                    ckas.append(torch.tensor(1.))
                else:
                    # ckas.append(compute_cka(activations[entry1], activations[entry2], device))
                    ckas.append(compute_minibatch_cka(activations[entry1], activations[entry2]))
    torch.cuda.empty_cache()
    return ckas, base


def row_cka(activations: dict, device: torch.device, permute: bool = False, n_permutations: int = 10):
    """
    Computes the CKA values for the activations passed in the given dictionary. Each entry in the dictionary contains
    another one that holds the activations for all models on a specific set of images. For each of these
    sub-dictionaries, the CKA values between the current model of interest and all other values are computed. Finally,
    the resulting list containing computed CKA values is returned.

    :param activations: The dictionary containing the activations.
    :param device: The device to which the activations should be transferred.
    :param permute: Whether to compute a baseline score on permuted inputs, default: False.
    :param n_permutations: The number of random permutations to compute for an average result, default: 10.
    :return: The list of computed CKA values.
    """
    ckas, base = [], []
    for entry1 in activations:
        row_activation = activations[entry1][entry1]
        for entry2 in tqdm.tqdm(activations[entry1], desc='| CKA for model {} |'.format(entry1),
                                total=len(activations[entry1])):
            if permute:
                mean_scores, scores = compute_permuted_cka(row_activation, activations[entry1][entry2], device,
                                                           n_permutations)
                ckas.append(mean_scores)
                base.append(scores)
            else:
                if entry1 == entry2:
                    ckas.append(torch.tensor(1.))
                else:
                    # ckas.append(compute_cka(row_activation, activations[entry1][entry2], device))
                    ckas.append(compute_minibatch_cka(row_activation, activations[entry1][entry2]))
    return ckas, base


def imshow(image: torch.Tensor, figsize: tuple = (10, 10)):
    """
    Displays the given tensor image as a figure with the given size.

    :param image: The tensor representation of the image.
    :param figsize: The size of the figure, default: (10, 10).
    """
    fig, ax = plt.subplots(figsize=figsize)
    image = image.numpy().transpose((1, 2, 0))
    ax.imshow(image)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    plt.grid(None)
    plt.show()


def get_seed_indices(labels: list, path: Optional[str] = None, lower_bound: int = 0, upper_bound: int = 49999,
                     dataset: str = 'imagenet', random_state: Optional[int]=None):
    """
    Generates a list of indices of images that can be used as seed images for the given list of labels of the target
    images. The labels of all available images are retrieved from the given file. For each label in the given list, a
    random number in the specified range is drawn. If the labels are the same, another random number is generated
    instead. In the end, a list of random numbers corresponding to the indices of the seed images to be used is
    returned.

    :param labels: The labels of the target images.
    :param path: The path to the labels file, default: None.
    :param lower_bound: The lower allowed bound for the random numbers, default: 0.
    :param upper_bound: The upper allowed bound for the random numbers, default: 49999.
    :param dataset: String indicating the dataset for which labels should be loaded, default: imagenet.
    :return: The indices of the seed images.
    """
    if random_state:
        random.seed(random_state)

    seed_indices = []
    # We use the labels of _all_ samples in the dataset to find a seed image with label distinct from the target image.
    if path is not None:
        if dataset == 'sat6':
            all_labels = np.argmax(pd.read_csv(path, header=None).values, 1)
        else:
            all_labels = np.array(pd.read_csv(path, index_col=0))
    else:
        all_labels = labels
    for label in labels:
        random_index = random.randint(lower_bound, upper_bound)
        while all_labels[random_index] == label:
            random_index = random.randint(lower_bound, upper_bound)
        seed_indices.append(random_index)
    return seed_indices


def get_inversions(model: AttackerModel, seed_images: torch.Tensor, target_images: torch.Tensor, batch_size: int = 32,
                   iterations: int = 1_000, device: torch.device = None):
    """
    Creates inverted images for the given model using the given seed and target images. The images are split into
    batches of the specified size and the inversion process is conducted using the given number of iterations.

    :param model: The model for which inverted images should be created.
    :param seed_images: The images to be used as seeds.
    :param target_images: The images to be used as targets.
    :param batch_size: The batch size for image inversion, default: 32.
    :param iterations: The number of iterations to perform, default: 1000.
    :param device: The device on which the inverted images should be generated.
    :return: The inverted images.
    """
    model.eval()
    inverted = invert_images(model, seed_images, target_images, batch_size, iterations=iterations, device=device)
    gc.collect()
    torch.cuda.empty_cache()
    return inverted


def save_inverted(file_path: str, inverted_images: torch.Tensor, save_as_uint8: bool=False):
    """
    Save the inverted images provided as tensors in the given file. If no file with the given name exists, a new file
    containing the images is created. Otherwise, the provided images are added to the ones already saved in the file.

    :param file_path: The path to the save file.
    :param inverted_images: The images to save.
    :param save_as_uint8: Whether to save in uint8 precision to save disk space. Ignored if the file_path already points
        to a uint8 tensor.
    """
    try:
        tensors = torch.load(file_path)
        if tensors.dtype == torch.uint8:
            logger.info("Saving inverted images in uint8 precision because prior tensors are already in uint8.")
            inverted_images = (255 * inverted_images).to(torch.uint8)
        concatenated = torch.cat((tensors, inverted_images), 0)
        torch.save(concatenated, file_path)
    except FileNotFoundError:
        if save_as_uint8:
            inverted_images = (255 * inverted_images).to(torch.uint8)
        torch.save(inverted_images, file_path)


def save_inverted_numpy(file_path: str, inverted_images: torch.Tensor):
    """
    Save the inverted images provided as tensors in the given file. If no file with the given name exists, a new file
    containing the images is created. Otherwise, the provided images are added to the ones already saved in the file.

    :param file_path: The path to the save file.
    :param inverted_images: The images to save.
    """
    images = inverted_images.numpy()
    try:
        array = np.load(file_path)
        concatenated = np.concatenate((array, images), 0)
        np.save(file_path, concatenated)
    except FileNotFoundError:
        np.save(file_path, images)


def sample_images(images: dict, num_samples: int = 1000, indices: list = None):
    """
    Draws a random subset of images from each entry in the given dictionary containing images for each model. If a list
    of indices is specified, images at these indices are taken instead of drawing randomly.

    :param images: The dictionary containing the images.
    :param num_samples: The number of images to be drawn, default: 1000.
    :param indices: The indices of the images to be selected, default: None.
    :return: The indices of the drawn images.
    """
    if indices is None:
        indices = random.sample(range(len(images[0])), num_samples)
    for entry in images:
        tensors = torch.index_select(images[entry], 0, torch.tensor(indices))
        images[entry] = tensors
    return indices
