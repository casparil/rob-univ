#!/bin/bash

echo "Assumes to be run in Max' rhaegal container 2"
echo "Adapt the device map with respect to currently free GPUs!"

device_map=(1 2 3 4 5)

dataPath="/root/univ-data/imagenet100/train.lmdb"
labelsPath="/root/univ-data/imagenet100/train_labels.csv"
outDir="/root/univ-data/imagenet100/inverted/"
seedIdxs="/root/univ-data/imagenet100/inverted/seed_indices.csv"
targetIdxs="/root/univ-data/imagenet100/inverted/target_indices.csv"
modelDir="/root/univ-data/imagenet100"

models=("densenet161" "resnet50" "tiny_vit_5m" "vgg16_bn" "wide_resnet50_2")

for i in {0..4}; do
    sessionName="inv-in100-$i"
    tmux kill-session -t "$sessionName"
    tmux new -s "$sessionName" -d
    tmux send-keys -t "$sessionName" "conda activate univ" Enter
    tmux send-keys -t "$sessionName" "clear" Enter

    # Every session deals with one model one their own GPU.
    model=${models[$i]}
    eps="eps3"
    tmux send-keys -t "$sessionName" "CUDA_VISIBLE_DEVICES=${device_map[$i]} python inversion.py --uint8 -d imagenet100 -e $eps -i $outDir -m $model -p imagenet100 -s $seedIdxs -t $targetIdxs --model-dir $modelDir/$eps/ --data-path $dataPath --labels-path $labelsPath && wait" Enter
    eps="eps1"
    tmux send-keys -t "$sessionName" "CUDA_VISIBLE_DEVICES=${device_map[$i]} python inversion.py --uint8 -d imagenet100 -e $eps -i $outDir -m $model -p imagenet100 -s $seedIdxs -t $targetIdxs --model-dir $modelDir/$eps/ --data-path $dataPath --labels-path $labelsPath && wait" Enter
    eps="eps05"
    tmux send-keys -t "$sessionName" "CUDA_VISIBLE_DEVICES=${device_map[$i]} python inversion.py --uint8 -d imagenet100 -e $eps -i $outDir -m $model -p imagenet100 -s $seedIdxs -t $targetIdxs --model-dir $modelDir/$eps/ --data-path $dataPath --labels-path $labelsPath && wait" Enter
    eps="eps025"
    tmux send-keys -t "$sessionName" "CUDA_VISIBLE_DEVICES=${device_map[$i]} python inversion.py --uint8 -d imagenet100 -e $eps -i $outDir -m $model -p imagenet100 -s $seedIdxs -t $targetIdxs --model-dir $modelDir/$eps/ --data-path $dataPath --labels-path $labelsPath && wait" Enter
    eps="eps0"
    tmux send-keys -t "$sessionName" "CUDA_VISIBLE_DEVICES=${device_map[$i]} python inversion.py --uint8 -d imagenet100 -e $eps -i $outDir -m $model -p imagenet100 -s $seedIdxs -t $targetIdxs --model-dir $modelDir/$eps/ --data-path $dataPath --labels-path $labelsPath && wait" Enter
done
