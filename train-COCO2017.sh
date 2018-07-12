#!/usr/bin/env bash

python3 train.py --train-path ./data/COCO2017/train.rec --val-path ./data/COCO2017/val.rec --network resnet50 --batch-size 16 --pretrained ./model/resnet-50 --epoch 0 --prefix ./output/exp-COCO-OHEM1/retina --gpus 0,1 --end-epoch 15 --data-shape 512 --label-width 600 --optimizer sgd --lr 0.01 --momentum 0.9 --wd 0.0001 --lr-steps '10, 13' --num-class 80 --class-names ./dataset/names/mscoco.names
