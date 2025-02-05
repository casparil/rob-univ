# Revisiting the Relation Between Robustness and Universality

This repository contains the implementation for the paper Revisiting the Relation Between Robustness and Universality.

## Setup

1. Create a virtual environment with Python `3.8.13` and activate it: `python -m venv .venv && source .venv/bin/activate`
2. Run `pip install ipykernel ipython lmdb==1.4.1 nltk==3.5 robustness==1.2.1.post2 tensorflow==2.13.1 timm==0.9.12 transformers==4.34.0 loguru` to install required packages.
3. Install cmake via `apt install cmake` aswell as `pip install git+https://github.com/simonzhang00/ripser-plusplus.git` and `pip install git+https://github.com/IlyaTrofimov/RTD.git`.
4. Download the necessary external data and organize it according to the folder structure described below.

## Data
To reproduce all paper results, the following data is required.
All data is distributed over four Zenodo datasets:

* https://zenodo.org/records/13936007
* https://zenodo.org/records/14802537
* https://zenodo.org/records/13936033
* https://zenodo.org/records/14802491


### Models

**ImageNet models:** Checkpoints for the pre-trained **l2-robust** CNNs with *eps=0* and *eps=3* are available
[here](https://huggingface.co/madrylab/robust-imagenet-models). Other models are on Zenodo.

**CIFAR-10 models:** Checkpoints for CIFAR-10 models are available from Zenodo.

### Datasets

**ImageNet:** It can be downloaded on the
[ImageNet website](https://www.image-net.org/) together with the ILSVRC2012 ImageNet validation set used for evaluation.

**CIFAR-10:** The CIFAR-10 dataset is used to train models from scratch and evaluate their similarity. It can be
downloaded [here](https://www.cs.toronto.edu/%7Ekriz/cifar.html).


### Inverted Images

The inverted images used for evaluation are available from Zenodo.

## Reproducing Results

### Model Training

#### ImageNet

To train a model on ImageNet, preprocess the ImageNet data into LMDB files using the `folder2lmdb.py` file contained in the
`utils` folder by running

```
# Creates train.lmdb file
python univ/utils/folder2lmdb.py -f data/imagenet/ILSVRC -s "train"

# Creates val.lmdb file
python univ/utils/folder2lmdb.py -f data/imagenet/ILSVRC -s "val"
```

To start the training, run

```
# Train standard ImageNet model, e.g. ResNet18
python imagenet_training.py -s ./data/cnns/imagenet/eps0/ -m resnet18

# Train standard ImageNet100 model, e.g. ResNet-18
python imagenet_training.py -s ./data/cnns/imagenet100/eps0/ -m resnet18 -n 100 -d ./data/imagenet/imagenet100/

# Train a robust ResNet18
python imagenet_training.py -s ./data/cnns/imagenet/eps3/ -m resnet18 -a 1
```

After training is finished, two checkpoint files `checkpoint.pt.latest` and `checkpoint.pt.best` will be available in
the specified save directory. To evaluate the pre-trained models on the validation set, run

```
# Evaluate robust ResNet18
python imagenet_training.py -t 0 -v ./data/cnns/imagenet/eps3/resnet18.ckpt
```

#### CIFAR-10

To train a CIFAR-10 model from scratch, run

```
# Train standard ResNet-18
python cifar10_training.py -m resnet18 -s ./data/cnns/cifar10/eps0/

# Train robust ResNet-18
python cifar10_training.py -m resnet18 -s ./data/cnns/cifar10/eps1/ -a 1
```

After training is finished, two checkpoint files `checkpoint.pt.latest` and `checkpoint.pt.best` will be available in
the specified save directory. To evaluate the pre-trained models on the test set, run

```
# Evaluate robust ResNet-18
python cifar10_training.py -t 0 -m resnet18 -v ./data/cnns/cifar10/eps1/resnet18.pt

# Evaluate standard ResNet-18
python cifar10_training.py -t 0 -m resnet18 -v ./data/cnns/cifar10/eps0/resnet18.pt
```

### Generating Inverted Images
Inverted images can be generated with `inversion.py`. See `scripts/invert_in100.sh` for an example.


### Model Similarity

Computing model similarity requires the pre-trained models along with the datasets described above as well as inverted
images to evaluate model feature usage. To calculate representational similarity run `rep.py`, see `scripts/compare_in100.sh` for an example.
For similarity of predictions, use `func.py`. See `scripts/comp_func_in100.sh` for an example.

### Similarity for Subgroups of the Data
To reproduce the results for Figure 5 of the paper, first use `subgroup_sim.py` to cache all comparison results.
Then use `notebooks/rep_vs_pred_sim.ipynb` to recreate the figure.

### Probing
Use `train_probe.py` to train the probes, and `compare_probes.py` to compare them.
See `scripts/extract_reps_for_probe_training.sh` for training across all models.
`notebooks/probe_agreement.ipynb` can be used to reproduce the Figure.


## Acknowledgements

The code of this project was developed with the help of the following repositories:

- https://github.com/implicitDeclaration/similarity
- https://haydn.fgl.dev/posts/a-better-index-of-similarity/
- https://github.com/js-d/sim_metric
- https://github.com/ahwillia/netrep
- https://github.com/MadryLab/robustness
- https://github.com/google-research/google-research/tree/master/do_wide_and_deep_networks_learn_the_same_things
- https://github.com/thecml/pytorch-lmdb
- https://github.com/dongxinshuai/RIFT-NeurIPS2021