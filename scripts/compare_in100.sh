#!/bin/bash

sessionName="comp-in100"

tmux kill-session -t $sessionName
tmux new -s $sessionName -d
tmux send-keys -t "$sessionName" "conda activate univ" Enter
tmux send-keys -t "$sessionName" "clear" Enter

epss=("eps0" "eps025" "eps05" "eps1" "eps3")
experimentId=10
modelBaseDir="/root/univ-data/imagenet100/"
indexFileDir="/root/thesis-code/results/indices/imagenet100/"
configFileDir="/root/thesis-code/results/configs/imagenet100/"
invImgsDir="/root/univ-data/imagenet100/inverted/"

for i in {0..4}; do
    eps=${epss[$i]}
    tmux send-keys -t "$sessionName" "CUDA_VISIBLE_DEVICES=6 python rep.py -r proc -d imagenet100 -i 1 -o $eps -e $experimentId --model-dir $modelBaseDir/$eps/ --index-file-dir $indexFileDir --inverted-imgs-dir $invImgsDir --models imagenet100 --knn 10 --config-file-path $configFileDir/imagenet100_$eps.json" Enter
done
