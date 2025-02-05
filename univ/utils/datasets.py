from loguru import logger


def get_cnn_attr(attr: dict, cnns: str, eps: str = 'eps3'):
    """
    Adds information specific to the given type of CNNs to the passed dictionary.

    :param attr: The dictionary.
    :param cnns: The type of CNNs to be used, i.e. ImageNet or CIFAR-10.
    :param eps: The type of robust models to use for ImageNet, default: eps3.
    :return: The updated dictionary.
    """
    assert eps in ['eps5', 'eps3', 'eps1', 'eps05', 'eps025', 'eps01', 'eps0']
    model_names = ['resnet18', 'resnet50', 'wide_resnet50_2', 'wide_resnet50_4', 'densenet161', 'resnext50_32x4d']

    if cnns == 'imagenet/':
        model_names.append('vgg16_bn')
        model_names.append('tiny_vit_5m')
        attr['eps'] = eps
        attr['standard_path'] = attr['cnn_path'] + 'eps0/'
        attr['robust_path'] = attr['cnn_path'] + eps + '/'
        if eps in ["eps0", 'eps3']:
            attr['model_names'] = model_names
            attr['layers'] = ['avgpool', 'avgpool', 'avgpool', 'avgpool', 'features.norm5', 'avgpool', 'classifier.5',
                              'head.drop']
            attr['four_dim_models'] = ['resnet18', 'resnet50', 'wide_resnet50_2', 'wide_resnet50_4', 'densenet161',
                                       'resnext50_32x4d']
        elif eps in ["eps025", "eps05", "eps1"]:
            attr['model_names'] = ['resnet18', 'resnet50', 'wide_resnet50_2', 'wide_resnet50_4', "tiny_vit_5m", "vgg16_bn"]
            attr['layers'] = ['avgpool', 'avgpool', 'avgpool', 'avgpool', 'head.drop', 'classifier.5']
            attr['four_dim_models'] = ['resnet18', 'resnet50', 'wide_resnet50_2', 'wide_resnet50_4']
        else:
            raise ValueError(f"Unknown epsilon level: {eps}")
    elif cnns == 'imagenet100/':
        # TODO: create TinyVit5m Checkpoints for IN100
        attr['model_names'] = ['resnet50', 'wide_resnet50_2', 'densenet161', 'vgg16_bn']
        # attr['model_names'] = ['resnet50', 'wide_resnet50_2', 'densenet161', 'vgg16_bn', 'tiny_vit_5m']
        attr['eps'] = eps
        attr['standard_path'] = attr['cnn_path'] + 'eps0/'
        attr['robust_path'] = attr['cnn_path'] + eps + '/'
        attr['layers'] = ['avgpool', 'avgpool', 'features.norm5', 'classifier.5']
        # attr['layers'] = ['avgpool', 'avgpool', 'features.norm5', 'classifier.5', 'head.drop']
        attr['four_dim_models'] = ['resnet50', 'wide_resnet50_2', 'densenet161']
    elif cnns == 'imagenet50/':
        # TODO: create TinyVit5m Checkpoints for IN50
        attr['model_names'] = ['resnet50', 'wide_resnet50_2', 'densenet161', 'vgg16_bn']
        # attr['model_names'] = ['resnet50', 'wide_resnet50_2', 'densenet161', 'vgg16_bn', 'tiny_vit_5m']
        attr['eps'] = eps
        attr['standard_path'] = attr['cnn_path'] + 'eps0/'
        attr['robust_path'] = attr['cnn_path'] + eps + '/'
        attr['layers'] = ['avgpool', 'avgpool', 'features.norm5', 'classifier.5']
        # attr['layers'] = ['avgpool', 'avgpool', 'features.norm5', 'classifier.5', 'head.drop']
        attr['four_dim_models'] = ['resnet50', 'wide_resnet50_2', 'densenet161']
    elif cnns == 'cifar10/':
        model_names.append('vgg16')
        attr['model_names'] = model_names
        attr['eps'] = eps
        attr['standard_path'] = attr['cnn_path'] + 'eps0/'
        attr['robust_path'] = attr['cnn_path'] + eps +'/'
        attr['layers'] = ['layer4', 'layer4', 'avgpool', 'avgpool', 'bn', 'avgpool', 'features.44']
        attr['four_dim_models'] = model_names
    else:
        raise NotImplementedError(f"{cnns=}")

    return attr


def get_dataset_attr(dataset: str = 'imagenet/', cnns: str = 'imagenet/', directory: str = 'thesis-code/',
                     eps: str = 'eps3'):
    """
    Constructs a dictionary containing information about dataset specific save locations, number of classes and model
    names. This method assumes a folder structure as presented in the README file.

    :param dataset: The dataset to be used, i.e. ImageNet, CIFAR-10, SAT-6 or SNLI default: imagenet/.
    :param cnns: The CNNs to be used, if the dataset is not SNLI, default: imagenet/.
    :param directory: The base directory containing the data and results folders.
    :param eps: The type of robust models to use for ImageNet, default: eps3.
    :return: A dictionary containing the dataset specific information.
    """
    attr = {
        'results_dir': directory + 'results/',
        'cnn_path': directory + 'data/cnns/' + cnns,
        'indices_path': directory + 'results/cka/inverted/10000/imagenet/target_indices_0.csv',
        'dict_paths': [directory + 'data/snli/dicts/bert.tokenized.dict.pkl',
                       directory + 'data/snli/dicts/roberta.tokenized.dict.pkl',
                       directory + 'data/snli/dicts/distilbert.tokenized.dict.pkl']
    }

    if dataset == 'snli/':
        attr['model_names'] = ['bert', 'roberta', 'distilbert']
        attr['num_classes'] = 3
        attr['standard_path'] = directory + 'data/lms/standard/'
        attr['robust_path'] = directory + 'data/lms/robust/'
        attr['data_path'] = directory + 'data/snli/split_snli_files.pkl'
        attr['labels_path'] = directory + 'data/snli/split_snli_files.pkl'
        attr['eps'] = 'rob'
        attr['save_std'] = directory + 'data/snli/inverted/{}.pkl'
        attr['save_rob'] = directory + 'data/snli/inverted/{}_rob.pkl'
        attr['layers'] = ['cls_to_logit.1', 'cls_to_logit.1', 'cls_to_logit.1']
        # attr['layers'] = ['plm.module.encoder.layer.11.output',
        #                  'plm.module.encoder.layer.11.output']
        attr['four_dim_models'] = []
    elif dataset == 'imagenet/':
        attr['num_classes'] = 1000
        attr['data_path'] = directory + 'data/imagenet/images/'
        attr['labels_path'] = directory + 'data/imagenet/labels/labels.csv'
        attr['save_std'] = directory + 'data/' + dataset + 'inverted/{}_eps0.pt'
        attr['save_rob'] = directory + 'data/' + dataset + 'inverted/{}_eps3.pt'
        attr['adv_std'] = directory + 'data/' + dataset + 'adv/{}_eps0.pt'
        attr['adv_rob'] = directory + 'data/' + dataset + 'adv/{}_' + eps + '.pt'
    elif dataset == 'imagenet100/':
        attr['num_classes'] = 100
        attr['data_path'] = directory + 'data/imagenet100/train.lmdb'
        attr['labels_path'] = ""
        attr['save_std'] = directory + 'data/' + dataset + 'inverted/{}_eps0.pt'
        attr['save_rob'] = directory + 'data/' + dataset + 'inverted/{}_' + eps + '.pt'
    elif dataset == 'imagenet50/':
        attr['num_classes'] = 50
        attr['data_path'] = directory + 'data/imagenet50/train.lmdb'
        attr['labels_path'] = ""
        attr['save_std'] = directory + 'data/' + dataset + 'inverted/{}_eps0.pt'
        attr['save_rob'] = directory + 'data/' + dataset + 'inverted/{}_' + eps + '.pt'
    elif dataset == 'cifar10/':
        attr['num_classes'] = 10
        attr['data_path'] = directory + 'data/cifar10/'
        attr['labels_path'] = directory + 'data/cifar10/'
        attr['save_std'] = directory + 'data/' + dataset + 'inverted/{}_eps0.npy'
        attr['save_rob'] = directory + 'data/' + dataset + 'inverted/{}_' + eps + '.npy'
    elif dataset == 'sat6/':
        attr['num_classes'] = 6
        attr['data_path'] = directory + 'data/sat6/X_test_sat6.csv'
        attr['labels_path'] = directory + 'data/sat6/y_test_sat6.csv'
        file_format = 'npy' if cnns == 'imagenet' else 'pt'
        attr['save_std'] = directory + 'data/' + dataset + 'inverted/' + cnns + '{}_eps0.' + file_format
        attr['save_rob'] = directory + 'data/' + dataset + 'inverted/' + cnns + '{}_' + eps + '.' + file_format
    else:
        raise NotImplementedError

    if dataset != 'snli/':
        attr = get_cnn_attr(attr, cnns, eps)

    return attr
