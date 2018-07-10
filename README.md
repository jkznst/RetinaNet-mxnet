# RetinaNet-mxnet
Adapted from [SSD](https://github.com/zhreshold/mxnet-ssd) implemented by zhreshold, the results still need to be tuned. Currently the master branch
 uses OHEM strategy, the FL is still under test.

### Differences from SSD
* We build FPN (P3 to P7) to replace the "multi_layer_feature" function;
* We build cls_subnet and bbox_subnet in the "multibox_layer" function, and the bias is initialized according to the [Focal Loss paper](https://arxiv.org/abs/1708.02002) (only for FL strategy);
* We use the anchor setting in the focal loss paper which is tested on COCO, but the best setting for PASCAL VOC still needs to be tuned;
* We adopt the [focal loss operator](https://github.com/eldercrow/focal_loss_mxnet_ssd) by eldercrow with small modification;
* We support converting COCO2017 data to rec format for training and validation.

### Usage

### mAP result
|    Model    |    Training data    |    Test data    |    Strategy    |    mAP    |    Note    |
|:----------------:|:---------------:|:------------:|:---------------:|:------:|:---------------|
| ResNet-50 512x512 | VOC07+12 trainval | VOC07 test | OHEM | 78.8 | sgd, lr0.01 |
| ResNet-50 512x512 | VOC07+12 trainval | VOC07 test | FL | 42.3 | adam, lr0.0005 |
