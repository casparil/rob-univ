#! /bin/bash

sessionName="rep-extraction"
tmux kill-session -t $sessionName
tmux new -s $sessionName -d
tmux send-keys -t $sessionName "conda activate univ" Enter
tmux send-keys -t $sessionName "clear" Enter

models=("densenet161" "resnet18" "vgg16_bn" "wide_resnet50_2" "wide_resnet50_4" "resnext50_32x4d")
for model in "${models[@]}"; do
    tmux send-keys -t $sessionName "python train_probe.py -m /root/univ-data/imagenet1k/eps3/$model.ckpt -a $model --dataset-name imagenet1k --batch-size 256 --workers 16 --epochs 0 --num-probes 0 --lmdb-file /root/univ-data/imagenet1k/train.lmdb --val-lmdb-file /root/univ-data/imagenet1k/val.lmdb --device cuda:6" Enter
    tmux send-keys -t $sessionName "mv reps.pt cache/reps_${model}_in1k_train.pt" Enter
    tmux send-keys -t $sessionName "mv val_reps.pt cache/reps_${model}_in1k_val.pt" Enter
done
