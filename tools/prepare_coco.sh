#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
mkdir ../data/COCO2017packed
python3 $DIR/prepare_dataset.py --dataset coco2017 --set train2017 --target $DIR/../data/COCO2017packed/train.lst  --root $DIR/../data/COCO2017
python3 $DIR/prepare_dataset.py --dataset coco2017 --set val2017 --target $DIR/../data/COCO2017packed/val.lst --shuffle False --root $DIR/../data/COCO2017
