#!/bin/bash
indexFileDir=/root/thesis-code/results/indices/imagenet100/
invertedImgsDir=/root/univ-data/imagenet100/inverted/
labelsPath=/root/univ-data/imagenet100/train_labels.csv
dataPath=/root/univ-data/imagenet100/train.lmdb
modelBaseDir=/root/univ-data/imagenet100/
gpu=0

sessionName=comp-f-in100
tmux kill-session -t $sessionName
tmux new -s $sessionName -d
tmux send-keys -t "$sessionName" "conda activate univ" Enter
tmux send-keys -t "$sessionName" "clear" Enter

# Disagreement
## Regular data
tmux send-keys -t "$sessionName" "CUDA_VISIBLE_DEVICES=$gpu python func.py --index-file-dir $indexFileDir --inverted-imgs-dir $invertedImgsDir --labels-path $labelsPath --data-path $dataPath --model-dir  $modelBaseDir/eps3/ --eps eps3 -m imagenet100 -f dis --exp 0 -d imagenet100" Enter
tmux send-keys -t "$sessionName" "CUDA_VISIBLE_DEVICES=$gpu python func.py --index-file-dir $indexFileDir --inverted-imgs-dir $invertedImgsDir --labels-path $labelsPath --data-path $dataPath --model-dir  $modelBaseDir/eps1/ --eps eps1 -m imagenet100 -f dis --exp 0 -d imagenet100" Enter
tmux send-keys -t "$sessionName" "CUDA_VISIBLE_DEVICES=$gpu python func.py --index-file-dir $indexFileDir --inverted-imgs-dir $invertedImgsDir --labels-path $labelsPath --data-path $dataPath --model-dir  $modelBaseDir/eps05/ --eps eps05 -m imagenet100 -f dis --exp 0 -d imagenet100" Enter
tmux send-keys -t "$sessionName" "CUDA_VISIBLE_DEVICES=$gpu python func.py --index-file-dir $indexFileDir --inverted-imgs-dir $invertedImgsDir --labels-path $labelsPath --data-path $dataPath --model-dir  $modelBaseDir/eps025/ --eps eps025 -m imagenet100 -f dis --exp 0 -d imagenet100" Enter
tmux send-keys -t "$sessionName" "CUDA_VISIBLE_DEVICES=$gpu python func.py --index-file-dir $indexFileDir --inverted-imgs-dir $invertedImgsDir --labels-path $labelsPath --data-path $dataPath --model-dir  $modelBaseDir/eps0/ --eps eps0 -m imagenet100 -f dis --exp 0 -d imagenet100" Enter

## Inverted data
tmux send-keys -t "$sessionName" "CUDA_VISIBLE_DEVICES=$gpu python func.py --index-file-dir $indexFileDir --inverted-imgs-dir $invertedImgsDir --labels-path $labelsPath --data-path $dataPath --model-dir  $modelBaseDir/eps3/ --eps eps3 -m imagenet100 -f dis --exp 0 -d imagenet100 -i 1" Enter
tmux send-keys -t "$sessionName" "CUDA_VISIBLE_DEVICES=$gpu python func.py --index-file-dir $indexFileDir --inverted-imgs-dir $invertedImgsDir --labels-path $labelsPath --data-path $dataPath --model-dir  $modelBaseDir/eps1/ --eps eps1 -m imagenet100 -f dis --exp 0 -d imagenet100 -i 1" Enter
tmux send-keys -t "$sessionName" "CUDA_VISIBLE_DEVICES=$gpu python func.py --index-file-dir $indexFileDir --inverted-imgs-dir $invertedImgsDir --labels-path $labelsPath --data-path $dataPath --model-dir  $modelBaseDir/eps05/ --eps eps05 -m imagenet100 -f dis --exp 0 -d imagenet100 -i 1" Enter
tmux send-keys -t "$sessionName" "CUDA_VISIBLE_DEVICES=$gpu python func.py --index-file-dir $indexFileDir --inverted-imgs-dir $invertedImgsDir --labels-path $labelsPath --data-path $dataPath --model-dir  $modelBaseDir/eps025/ --eps eps025 -m imagenet100 -f dis --exp 0 -d imagenet100 -i 1" Enter
tmux send-keys -t "$sessionName" "CUDA_VISIBLE_DEVICES=$gpu python func.py --index-file-dir $indexFileDir --inverted-imgs-dir $invertedImgsDir --labels-path $labelsPath --data-path $dataPath --model-dir  $modelBaseDir/eps0/ --eps eps0 -m imagenet100 -f dis --exp 0 -d imagenet100 -i 1" Enter

# JSD
## Regular data
tmux send-keys -t "$sessionName" "CUDA_VISIBLE_DEVICES=$gpu python func.py --index-file-dir $indexFileDir --inverted-imgs-dir $invertedImgsDir --labels-path $labelsPath --data-path $dataPath --model-dir  $modelBaseDir/eps3/ --eps eps3 -m imagenet100 -f jsd --exp 0 -d imagenet100" Enter
tmux send-keys -t "$sessionName" "CUDA_VISIBLE_DEVICES=$gpu python func.py --index-file-dir $indexFileDir --inverted-imgs-dir $invertedImgsDir --labels-path $labelsPath --data-path $dataPath --model-dir  $modelBaseDir/eps1/ --eps eps1 -m imagenet100 -f jsd --exp 0 -d imagenet100" Enter
tmux send-keys -t "$sessionName" "CUDA_VISIBLE_DEVICES=$gpu python func.py --index-file-dir $indexFileDir --inverted-imgs-dir $invertedImgsDir --labels-path $labelsPath --data-path $dataPath --model-dir  $modelBaseDir/eps05/ --eps eps05 -m imagenet100 -f jsd --exp 0 -d imagenet100" Enter
tmux send-keys -t "$sessionName" "CUDA_VISIBLE_DEVICES=$gpu python func.py --index-file-dir $indexFileDir --inverted-imgs-dir $invertedImgsDir --labels-path $labelsPath --data-path $dataPath --model-dir  $modelBaseDir/eps025/ --eps eps025 -m imagenet100 -f jsd --exp 0 -d imagenet100" Enter
tmux send-keys -t "$sessionName" "CUDA_VISIBLE_DEVICES=$gpu python func.py --index-file-dir $indexFileDir --inverted-imgs-dir $invertedImgsDir --labels-path $labelsPath --data-path $dataPath --model-dir  $modelBaseDir/eps0/ --eps eps0 -m imagenet100 -f jsd --exp 0 -d imagenet100" Enter

## Inverted data
tmux send-keys -t "$sessionName" "CUDA_VISIBLE_DEVICES=$gpu python func.py --index-file-dir $indexFileDir --inverted-imgs-dir $invertedImgsDir --labels-path $labelsPath --data-path $dataPath --model-dir  $modelBaseDir/eps3/ --eps eps3 -m imagenet100 -f jsd --exp 0 -d imagenet100 -i 1" Enter
tmux send-keys -t "$sessionName" "CUDA_VISIBLE_DEVICES=$gpu python func.py --index-file-dir $indexFileDir --inverted-imgs-dir $invertedImgsDir --labels-path $labelsPath --data-path $dataPath --model-dir  $modelBaseDir/eps1/ --eps eps1 -m imagenet100 -f jsd --exp 0 -d imagenet100 -i 1" Enter
tmux send-keys -t "$sessionName" "CUDA_VISIBLE_DEVICES=$gpu python func.py --index-file-dir $indexFileDir --inverted-imgs-dir $invertedImgsDir --labels-path $labelsPath --data-path $dataPath --model-dir  $modelBaseDir/eps05/ --eps eps05 -m imagenet100 -f jsd --exp 0 -d imagenet100 -i 1" Enter
tmux send-keys -t "$sessionName" "CUDA_VISIBLE_DEVICES=$gpu python func.py --index-file-dir $indexFileDir --inverted-imgs-dir $invertedImgsDir --labels-path $labelsPath --data-path $dataPath --model-dir  $modelBaseDir/eps025/ --eps eps025 -m imagenet100 -f jsd --exp 0 -d imagenet100 -i 1" Enter
tmux send-keys -t "$sessionName" "CUDA_VISIBLE_DEVICES=$gpu python func.py --index-file-dir $indexFileDir --inverted-imgs-dir $invertedImgsDir --labels-path $labelsPath --data-path $dataPath --model-dir  $modelBaseDir/eps0/ --eps eps0 -m imagenet100 -f jsd --exp 0 -d imagenet100 -i 1" Enter
