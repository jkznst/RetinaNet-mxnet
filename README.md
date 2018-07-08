# RetinaNet-mxnet
Adapted from SSD implemented by zhreshold, still under construction.

Currently the sizes of the anchor boxes are different from the FocalLoss paper, due to the use of mx.contrib.MultiBoxPrior function.

### mAP result
|    Model    |    Training data    |    Test data    |    Strategy    |    mAP    |
|:----------------:|:---------------:|:------------:|:---------------:|:------|
| [ResNet-50 512x512] | VOC07+12 trainval | VOC07 test | OHEM | 78.8 |
