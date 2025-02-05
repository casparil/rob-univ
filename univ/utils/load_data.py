import math
import numpy as np
import os
import os.path
import pandas as pd
import torch
import tqdm
import random

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder
from univ.rift.data import SnliData
from univ.rift import tokenizers
from univ.rob import datasets
from univ.utils.folder2lmdb import ImageFolderLMDB
from robustness import data_augmentation as da
from loguru import logger

from typing import List, Dict, Union, Optional, Any

try:
    import cPickle as pickle
except ImportError:
    import pickle


class ImageLoader(Dataset):
    """
    A dataloader class for loading images and their corresponding labels.
    """

    def __init__(self, labels_file: str, root_dir: str, transform: any = None, num_images: int = 10000,
                 indices: list = None):
        """
        Randomly samples 10000 indices representing images and labels to be loaded from the specified folders. The
        number of images is expected to be 50000 as these are the number of images contained in the ImageNet validation
        set. After the indices have been sampled, the corresponding labels and file names of the images are retrieved.

        :param labels_file: The path to the file containing the labels.
        :param root_dir: The path to the directory containing the ImageNet pictures.
        :param transform: A list of transformations to be performed on each image, should transform it to a tensor,
        default: None.
        :param num_images: The number of images to sample, default: 10000.
        """
        if indices is None:
            self.indices = random.sample(range(50000), num_images)
        else:
            self.indices = indices
        self.labels = np.array(pd.read_csv(labels_file, header=None, names=['y'])).take(self.indices)
        self.root_dir = root_dir
        self.transform = transform
        images = os.listdir(self.root_dir)
        images.sort()
        self.images = [images[idx] for idx in self.indices]

    def __len__(self):
        """
        Returns the total number of images to be loaded.

        :return: The number of images.
        """
        return len(self.images)

    def __getitem__(self, idx: int):
        """
        Retrieves an image at the specified index from the image folder, transforms it, and returns it together with its
        label.

        :param idx: The index.
        :return: The tensor image together with its label.
        """
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')
        tensor_image = self.transform(image)
        return tensor_image, self.labels[idx]


class ImageLoaderLMDB(Dataset):
    def __init__(self, lmdb_dataset: ImageFolderLMDB, indices: List[int]):
        self.dataset = lmdb_dataset
        self.indices = indices
        self.local_idx_to_lmdb_idx = {i: indices[i] for i in range(len(indices))}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.dataset[self.local_idx_to_lmdb_idx[idx]]



class CsvImageLoader(Dataset):
    """
    A dataloader class for loading images for SAT-6 and their corresponding labels.
    """

    def __init__(self, labels_file: str, images_path: str, transform: any = None, num_images: int = 10000,
                 indices: list = None):
        """
        Randomly samples 10000 indices representing images and labels to be loaded from the specified folders. The
        number of images is expected to be 81000 as these are the number of images contained in the SAT-6 test set.
        After the indices have been sampled, the corresponding labels and images are retrieved.

        :param labels_file: The path to the file containing the labels.
        :param images_path: The path to the file containing the pictures.
        :param transform: A list of transformations to be performed on each image, should transform it to a tensor,
        default: None.
        :param num_images: The number of images to sample, default: 10000.
        """
        if indices is None:
            self.indices = random.sample(range(81000), num_images)
        else:
            self.indices = indices
        self.labels = np.argmax(pd.read_csv(labels_file, header=None).values, 1).take(self.indices)
        self.images_dir = images_path
        self.transform = transform
        self.images = np.array(pd.read_csv(images_path, header=None).values.reshape((-1, 28, 28, 4)))[self.indices]

    def __len__(self):
        """
        Returns the total number of images to be loaded.

        :return: The number images.
        """
        return len(self.images)

    def __getitem__(self, idx: int):
        """
        Retrieves an image at the specified index, transforms it, and returns it together with its label.

        :param idx: The index.
        :return: The tensor image together with its label.
        """
        image = Image.fromarray(self.images[idx].astype('uint8'), 'RGBA').convert('RGB')
        tensor_image = self.transform(image)
        return tensor_image, self.labels[idx]


def load_images(labels_file: str, root_dir: str, transform: any = None, num_samples: int = 10, indices: list = None):
    """
    Loads images from the specified directory and transforms them to tensors. If a list of indices representing indexes
    of images to be loaded is provided, these images are loaded. If not, the specified number of sample indices is drawn
    instead.

    :param labels_file: The path to the file containing the image labels.
    :param root_dir: The path to the directory containing the ImageNet pictures.
    :param transform: A list of transformations to be performed on each image, default: None.
    :param num_samples: The number of random images to be loaded, ignored if a list of indices is provided, default: 10.
    :param indices: A list of numbers specifying which images to load, default: None.
    :return: A list of images and labels.
    """
    if indices is None:
        indices = random.sample(range(50000), num_samples)
    img_names = os.listdir(root_dir)
    img_names.sort()
    print(img_names)
    img_names = [img_names[idx] for idx in indices]
    images = []
    labels = np.array(pd.read_csv(labels_file, index_col=0)).take(indices)
    for file in tqdm.tqdm(img_names, desc='| Loading images |', total=(len(img_names))):
        img_name = os.path.join(root_dir, file)
        image = Image.open(img_name).convert('RGB')

        if transform is not None:
            image = transform(image)

        images.append(image)
    return images, labels


def load_sat6_images(labels_file: str, images_file: str, transform: any = None, indices: list = None,
                     num_samples: int = 10):
    """
    Loads the SAT-6 images and their labels at the given indices and transforms them according to the given rules. If no
    indices are specified, random samples are drawn.

    :param labels_file: The path to the file containing image labels.
    :param images_file: The path to the file containing image data.
    :param transform: A list of transformations to be performed on each image, default: None.
    :param indices: A list of numbers specifying which images to load, default: None.
    :param num_samples: The number of random images to be loaded, ignored if a list of indices is provided, default: 10.
    :return: A list of images and labels.
    """
    if indices is None:
        indices = random.sample(range(81000), num_samples)
    labels = np.argmax(pd.read_csv(labels_file, header=None).values, 1).take(indices)
    samples = np.array(pd.read_csv(images_file, header=None).values.reshape((-1, 28, 28, 4)))[indices]
    images = transform_images(samples, transform)
    return images, labels


def load_imagenet100_or_50_images(root_dir: str, transform: any = None, indices: list = None, num_samples: int = 10):
    """
    Loads the ImageNet100 images and their labels at the given indices and transforms them to tensors. If no indices are
    specified, random samples are drawn.

    :param root_dir: The path to the directory containing ImageNet100 training images.
    :param transform: A list of transformations to be performed on each image, default: None.
    :param indices: A list of numbers specifying which images to load, default: None.
    :param num_samples: The number of random images to be loaded, ignored if a list of indices is provided, default: 10.
    :return: A list of images and labels.
    """
    if indices is None:
        indices = random.sample(range(130000), num_samples)

    def raw_reader(path):
        with open(path, 'rb') as f:
            bin_data = f.read()
        return bin_data

    if os.path.isdir(root_dir):
        folder = ImageFolder(root_dir, transform=transform, loader=raw_reader)
    elif root_dir.endswith(".lmdb"):
        folder = ImageFolderLMDB(root_dir, transform=transform)
    else:
        raise ValueError(f"Path to dataset ({root_dir}) points to an unknown file.")
    images, labels = [], []

    for idx in indices:
        x, y = folder.__getitem__(idx)
        images.append(x)
        labels.append(y)

    return images, labels


def load_inverted_images(model_names: List[str], inv_imgs_dir: str, eps: str) -> Dict[int, torch.Tensor]:
    """
    Loads inverted images saved as torch tensors or numpy arrays named after the models provided in the list from the
    specified path.

    :param model_names: The names of the models.
    :param inv_imgs_dir: The path where the data is stored.
    :return: A dictionary containing an entry for each model with its corresponding inverted images.
    """
    images = {}
    for idx, name in tqdm.tqdm(enumerate(model_names), desc='| Load inverted images |', total=len(model_names)):
        load_path = os.path.join(inv_imgs_dir, f"{name}_{eps}.pt")
        if load_path.endswith('.pt'):
            images[idx] = torch.load(load_path)
            # Neural networks are not scale invariant.
            # Thus, we need to scale the uint8-formatted tensors (in range [0, 255]) back to the [0,1] range.
            if images[idx].dtype == torch.uint8:
                images[idx] = images[idx].float() / 255
        elif load_path.endswith('.npy'):
            img = np.load(load_path)
            images[idx] = torch.from_numpy(img)
        else:
            raise ValueError(f"No inverted images found at {load_path}.")
    return images


def save_indices(indices: list, path: str):
    """
    Saves the given list of image indices as a pandas dataframe under the specified path.

    :param indices: The list indices to be saved.
    :param path: The path and filename where the dataframe should be saved.
    """
    df = pd.DataFrame(np.array(indices))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)


def transform_cifar_images(images: list, transform: any):
    """
    Transforms the given images according to the provided transformation rules and returns the transformed images.

    :param images: The initial CIFAR-10 images.
    :param transform: A list of transformations to be performed on each image.
    """
    transformed = []
    for image in images:
        image = Image.fromarray(image)
        transformed.append(transform(image))
    return transformed


def transform_images(samples: np.ndarray, transform: any = None):
    """
    Transforms the given images to tensors, so they can be displayed.

    :param samples: The images to transform.
    :param transform: A list of transformations to be performed on each image, default: None.
    :return: A list of transformed images.
    """
    images = []
    for sample in tqdm.tqdm(samples, desc='| Loading images |', total=(len(samples))):
        image = Image.fromarray(sample.astype('uint8'), 'RGBA').convert('RGB')

        if transform is not None:
            image = transform(image)

        images.append(image)
    return images


def get_image_data(dataset: str, labels_path: str, image_path: str, indices_path: str, batch_size: int = 64,
                   num_samples: int = 10, use_cifar: bool = False):
    """
    Retrieves a set of 10,000 images and labels for the given dataset and creates a dataloader for them. For the
    ImageNet and SAT-6 datasets, a random subset of 10,000 images is drawn. For CIFAR-10, the test batch is used.

    :param dataset: The dataset for which the data should be retrieved, i.e. ImageNet, SAT-6 or CIFAR-10.
    :param labels_path: The path to the image labels for loading ImageNet or SAT-6 data.
    :param image_path: The path to the image data.
    :param indices_path: The path where the indices of the used data points should be saved or the path to a file
    containing indices to be loaded.
    :param batch_size: The batch size to use, default: 64.
    :param num_samples: The number of sample images to load, so they can be displayed, default: 10.
    :param use_cifar: Boolean indicating whether images should be scaled to CIFAR-10 or ImageNet size for SAT-6,
    default: False.
    :return: The loaded labels and images along with the created dataloader.
    """
    indices = None
    if os.path.isfile(indices_path):
        indices = list(pd.read_csv(indices_path, index_col=0)['0'])
    if dataset in ['imagenet', "imagenet100", "imagenet50"]:
        transform = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()])
        if os.path.isdir(image_path):
            data_sample = ImageLoader(labels_file=labels_path, root_dir=image_path, transform=transform, indices=indices)
            images, labels = load_images(labels_path, image_path, transform, num_samples, data.indices[0:num_samples])
            if indices is None:
                save_indices(data.indices, indices_path)
        elif image_path.endswith(".lmdb"):
            data = ImageFolderLMDB(image_path, transform=transform)
            if indices is None:
                n_val_imgs_in1k = 50000
                imgs_to_use = num_samples
                indices = random.sample(range(n_val_imgs_in1k), imgs_to_use)
                save_indices(indices, indices_path)
            data_sample = ImageLoaderLMDB(data, indices)

            images, labels = [], []
            for idx in tqdm.tqdm(indices, desc='| Loading images |'):
                img, label = data[idx]
                images.append(img)
                labels.append(label)
        else:
            raise ValueError(f"Cannot load images from {image_path=}")

        dataloader = DataLoader(data_sample, batch_size=batch_size)
    elif dataset == 'sat6':
        transform = da.TEST_TRANSFORMS_DEFAULT(32) if use_cifar else transforms.Compose([transforms.Resize(255),
                                                                                         transforms.CenterCrop(224),
                                                                                         transforms.ToTensor()])
        data = CsvImageLoader(labels_file=labels_path, images_path=image_path, transform=transform, indices=indices)
        images = transform_images(data.images[0:num_samples], transform)
        labels = data.labels
        dataloader = DataLoader(data, batch_size=batch_size)
        if indices is None:
            save_indices(data.indices, indices_path)
    elif dataset == 'cifar10':
        ds = CIFAR10(root=image_path, train=False, transform=da.TEST_TRANSFORMS_DEFAULT(32), download=True)
        labels = ds.targets
        images = transform_cifar_images(ds.data, da.TEST_TRANSFORMS_DEFAULT(32))
        data_path = os.path.expandvars(image_path)
        dataset = datasets.DATASETS['cifar'](data_path)
        _, dataloader = dataset.make_loaders(4, batch_size, data_aug=False)
    else:
        raise NotImplementedError
    return images, labels, [dataloader]


def read_snli_train_data(data_path: str, indices: Optional[list] = None):
    """
    Retrieves the SNLI training data from the pre-processed file located in the given directory and returns a subset of
    it specified by the given list of indices, if present.

    :param data_path: The path to the SNLI file.
    :param indices: A list of indices specifying which premise-hypothesis pairs and their labels should be returned.
    :return: The subset of premise-hypothesis pairs along with their labels.
    """
    if os.path.exists(data_path):
        print('Read processed SNLI dataset')
        with open(data_path, 'rb') as f:
            saved = pickle.load(f)
        train_perms = np.array(saved['train_perms'])
        train_hypos = np.array(saved['train_hypos'])
        train_labels = np.array(saved['train_labels'])
    else:
        raise FileNotFoundError
    if indices is not None:
        return train_perms[indices].tolist(), train_hypos[indices].tolist(), train_labels[indices]
    else:
        return train_perms.tolist(), train_hypos.tolist(), train_labels


def get_snli_data(dict_path: str, perms: list, hypos: list, labels: list, batch_size: int = 64, llm: str = 'bert',
                  return_data: bool = False):
    """
    Retrieves the SNLI data and constructs a dataloader for the give language model.

    :param dict_path: The path to the model's dictionary file.
    :param perms: A list of premises.
    :param hypos: A list of hypotheses.
    :param labels: A list of labels for the premise-hypothesis pairs.
    :param batch_size: The batch size for the dataloader.
    :param llm: The type of language model for which a dataloader should be constructed.
    :param return_data: Boolean indicating whether the full data or only the dataloader should be returned.
    """
    if os.path.exists(dict_path):
        with open(dict_path, 'rb') as f:
            saved = pickle.load(f)
        tokenized_subs_dict = saved["tokenized_subs_dict"]
    else:
        raise FileNotFoundError
    tokenizer, substitution_tokenizer = tokenizers.get_tokenizers(llm)
    opt = {
        'plm_type': llm
    }
    data = SnliData(opt, perms, hypos, labels, tokenized_subs_dict, 80, tokenizer, given_class=None)
    loader = DataLoader(data, batch_size, shuffle=False)
    if return_data:
        return (perms, hypos), labels, loader
    else:
        return loader


def get_data(dataset: str, labels_path: str, data_path: str, indices_path: str, dict_paths: list, batch_size: int = 64,
             num_samples: int = 10, use_cifar: bool = False):
    """
    Retrieves a set of 10,000 images or premise-hypothesis pairs and labels for the given dataset and creates a
    dataloader for them. For the ImageNet, SAT-6 and SNLI datasets, a random subset of 10,000 data instances is drawn.
    For CIFAR-10, the test batch is used.

    :param dataset: The dataset for which the data should be retrieved, i.e. ImageNet, SAT-6, CIFAR-10 or SNLI.
    :param labels_path: The path to the image labels for loading ImageNet or SAT-6 data.
    :param data_path: The path to the data.
    :param indices_path: The path where the indices of the used ImageNet, SAT-6 or SNLI data should be saved or the path
    to a file containing indices to be loaded.
    :param dict_paths: A path to the dictionaries for the language models when loading SNLI data.
    :param batch_size: The batch size to use, default: 64.
    :param num_samples: The number of sample images to load, so they can be displayed, default: 10.
    :param use_cifar: Boolean indicating whether images should be scaled to CIFAR-10 or ImageNet size for SAT-6,
    default: False.
    :return: The loaded labels and images along with the created dataloader.
    """
    logger.info("Loading regular images")

    if dataset == 'snli':
        if os.path.isfile(indices_path):
            indices = list(pd.read_csv(indices_path, index_col=0)['0'])
        else:
            indices = random.sample(range(549367), 10000)
            save_indices(indices, indices_path)
        archs = ['bert', 'roberta', 'distilbert']
        assert len(dict_paths) == len(archs)
        data, labels = None, None
        loaders = []
        perms, hypos, y = read_snli_train_data(data_path, indices)
        for idx, arch in enumerate(archs):
            if idx == 0:
                data, labels, loader = get_snli_data(dict_paths[idx], perms, hypos, y, batch_size, arch, True)
            else:
                loader = get_snli_data(dict_paths[idx], perms, hypos, y, batch_size, arch)
            loaders.append(loader)
        return data, labels, loaders
    else:
        return get_image_data(dataset, labels_path, data_path, indices_path, batch_size, num_samples, use_cifar)


def load_inverted_data(model_names: list, path: str, dataset: str, eps: str) -> Union[Dict[int, List[Any]], Dict[int, torch.Tensor]]:
    """
    Loads inverted data for the given dataset and returns it in a dictionary.

    :param model_names: The names of the models.
    :param path: The path where the data is stored.
    :param dataset: The dataset for which the data should be loaded.
    :return: A dictionary containing the data.
    """
    if dataset == 'snli':
        data = {}
        for idx, model in enumerate(model_names):
            perms, hypos, labels = read_snli_train_data(os.path.join(path, f"{model}_{eps}.pt"), None)
            data[idx] = {0: perms, 1: hypos, 2: labels}
        return data
    else:
        return load_inverted_images(model_names, path, eps)


def save_inverted_snli_data(stats_path: str, data_path: str, tested_num: int, success_num: int, fail_num: int,
                            sub_rates: float, prems: list, hypos: list, labels: list):
    """
    Saves the given data generated by computing adversarial example in two dataframes, one for the premise-hypothesis
    pairs along with their labels under the given data path and another one containing the given statistics under the
    stats path. If both files already exist, the current SNLI data is appended to the old one and the statistics are
    updated before saving.

    :param stats_path: The path where the statistics should be saved.
    :param data_path: The path where the SNLI data should be saved.
    :param tested_num: The number of attempts made to generate adversarial examples.
    :param success_num: The number of generated adversarial examples.
    :param fail_num: The number of failed attempts.
    :param sub_rates: The substitution rate summed up over all attempts.
    :param prems: The adversarial premises.
    :param hypos: The adversarial hypotheses.
    :param labels: The ground truth labels.
    """
    if os.path.isfile(data_path) and os.path.isfile(stats_path):
        data_df = pd.read_csv(data_path, index_col=0)
        stats_df = pd.read_csv(stats_path, index_col=0)
        new_df = pd.DataFrame()
        new_df['premise'], new_df['hypos'], new_df['labels'] = prems, hypos, labels
        data_df = pd.concat([data_df, new_df], ignore_index=True)
        stats_df['tested'] = stats_df['tested'] + tested_num
        stats_df['success'] = stats_df['success'] + success_num
        stats_df['fail'] = stats_df['fail'] + fail_num
        stats_df['sub_rates'] = stats_df['sub_rates'] + sub_rates
    else:
        data_df = pd.DataFrame()
        data_df['premise'], data_df['hypos'], data_df['labels'] = prems, hypos, labels
        stats_df = pd.DataFrame()
        stats_df['tested'] = [tested_num]
        stats_df['success'], stats_df['fail'], stats_df['sub_rates'] = success_num, fail_num, sub_rates
    data_df.to_csv(data_path)
    stats_df.to_csv(stats_path)


def subsample_adversarial_snli_data(data: Dict[int, torch.Tensor], indices: list, subsampled_data: Union[Dict[int, np.ndarray], None]):
    """
    Takes a subsample at the given indices of the standard and robust data expected to contain adversarial SNLI examples
    for a non-robust and robust model, respectively. The sampled data is then added to the given subsampled data, if it
    has previously been initialized.

    :param standard_data: Tensor containing adversarial examples of a non-robust model.
    :param robust_data: Tensor containing adversarial examples of a robust model.
    :param indices: A list of indices for sampling the data.
    :param subsampled_data: Dictionary containing previously sampled data, if it has been initialized.
    :return: A dictionary containing the sampled adversarial examples.
    """
    prems = np.array(data[0]).take(indices)
    hypos = np.array(data[1]).take(indices)
    labels = np.array(data[2]).take(indices)
    if subsampled_data is None:
        subsampled_data = {0: prems, 1: hypos, 2: labels}
    else:
        subsampled_data[0] = np.concatenate((subsampled_data[0], prems), axis=0)
        subsampled_data[1] = np.concatenate((subsampled_data[1], hypos), axis=0)
        subsampled_data[2] = np.concatenate((subsampled_data[2], labels), axis=0)
    return subsampled_data


def subsample_inverted_images(inverted_imgs: torch.Tensor, indices: list, subsampled_data: Union[torch.Tensor, None]):
    """
    Takes a subsample at the given indices of the standard and robust data expected to contain inverted images for a
    non-robust and robust model, respectively. The sampled images are then added to the given subsampled data, if it has
    already been initialized.

    :param standard_data: Tensor containing inverted images of a non-robust model.
    :param robust_data: Tensor containing inverted images of a robust model.
    :param indices: A list of indices for sampling the data.
    :param subsampled_data: Tensor containing previously sampled data, if it has been initialized.
    :return: A tensor containing the sampled inverted images.
    """
    sampled_inverted_imgs = torch.index_select(inverted_imgs, 0, torch.tensor(indices))
    if subsampled_data is None:
        subsampled_data = sampled_inverted_imgs
    else:
        subsampled_data = torch.cat((subsampled_data, sampled_inverted_imgs), 0)
    return subsampled_data


def subsample_inverted_data(data: Dict[int, Any], path_to_index_file: str, sample_size: int = 10000,
                            max_index: int = 10000, dataset: str = 'imagenet/'):
    """
    Random data samples are taken from each entry in the given dictionaries expected to contain inverted images or
    adversarial SNLI examples. The number of samples taken per entry depends on the given sample size indicating the
    desired total number of examples to be returned.

    :param standard_data: Dictionary containing data of non-robust models.
    :param robust_data: Dictionary containing data of robust models.
    :param path: The path where the indices of the used data points should be saved or the path to a file containing
    indices to be loaded.
    :param sample_size: The desired total number of data points to be returned, default: 10,000.
    :param max_index: The number of datapoints from which to sample
    :param dataset: The dataset for which should be subsampled, default: imagenet/.
    :return: The sampled data and sample indices.
    """
    subsampled_data, loaded_indices = None, None
    num_samples = math.ceil(sample_size / (len(data) * 2))
    all_indices = []
    if os.path.isfile(path_to_index_file):
        loaded_indices = list(pd.read_csv(path_to_index_file, index_col=0)['0'])
    for idx, _ in enumerate(data):
        start = idx * num_samples
        end = start + num_samples
        if dataset == 'snli/':
            if loaded_indices is None:
                indices = random.sample(range(max_index), num_samples)
            else:
                indices = loaded_indices[start:end]
            subsampled_data = subsample_adversarial_snli_data(data[idx], indices, subsampled_data)
        else:
            if loaded_indices is None:
                indices = random.sample(range(max_index), num_samples)
            else:
                indices = loaded_indices[start:end]
            assert isinstance(subsampled_data, (torch.Tensor, type(None)))
            subsampled_data = subsample_inverted_images(data[idx], indices, subsampled_data)
        all_indices += indices.copy()
    if dataset == 'snli/':
        assert isinstance(subsampled_data, dict)
        subsampled_data[0] = subsampled_data[0].tolist()
        subsampled_data[1] = subsampled_data[1].tolist()
    data = {}
    for num in range(len(data)):
        data[num] = subsampled_data
    if loaded_indices is None:
        save_indices(all_indices, path_to_index_file)
    return data, all_indices


def sample_labels(labels: np.ndarray, indices: list, sample_size: int, num_models: int):
    """
    Samples the labels for sub-sampled inverted images at the given and returns them.

    :param labels: A list of labels to sample.
    :param indices: A list containing indices inverted images that were sampled.
    :param sample_size: The number of sampled images.
    :param num_models: The number of models for which images were sampled.
    :return: A list of labels matching the sampled images.
    """
    num_samples = math.ceil(sample_size / num_models)
    all_labels = None
    for num in range(num_models):
        start = num * num_samples
        end = start + num_samples
        samples = np.concatenate((labels.take(indices[start:end]), labels.take(indices[start:end])))
        if all_labels is None:
            all_labels = samples
        else:
            all_labels = np.concatenate((all_labels, samples))
    return all_labels
