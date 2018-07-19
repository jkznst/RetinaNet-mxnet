#!/usr/bin/env bash

python3 train.py --train-path ./data/VOCpacked/train.rec --val-path ./data/VOCpacked/val.rec --network resnet50 --batch-size 16 --pretrained ./model/resnet-50 --epoch 0 --prefix ./output/exp-VOC-FL/retina --gpus 2,3 --end-epoch 90 --data-shape 512 --label-width 350 --optimizer sgd --lr 0.01 --momentum 0.9 --wd 0.0001 --lr-steps '60, 80' --num-class 20 --class-names ./dataset/names/pascal_voc.names
