import mxnet as mx
import numpy as np

def conv_act_layer(from_layer, name, num_filter, kernel=(1,1), pad=(0,0), \
    stride=(1,1), act_type="relu", use_batchnorm=False):
    """
    wrapper for a small Convolution group

    Parameters:
    ----------
    from_layer : mx.symbol
        continue on which layer
    name : str
        base name of the new layers
    num_filter : int
        how many filters to use in Convolution layer
    kernel : tuple (int, int)
        kernel size (h, w)
    pad : tuple (int, int)
        padding size (h, w)
    stride : tuple (int, int)
        stride size (h, w)
    act_type : str
        activation type, can be relu...
    use_batchnorm : bool
        whether to use batch normalization

    Returns:
    ----------
    (conv, relu) mx.Symbols
    """
    bias = mx.symbol.Variable(name="{}_conv_bias".format(name),   
        init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
    conv = mx.symbol.Convolution(data=from_layer, kernel=kernel, pad=pad, \
        stride=stride, num_filter=num_filter, name="{}_conv".format(name), bias=bias)
    if use_batchnorm:
        conv = mx.symbol.BatchNorm(data=conv, name="{}_bn".format(name))
    relu = mx.symbol.Activation(data=conv, act_type=act_type, \
        name="{}_{}".format(name, act_type))
    return relu

def legacy_conv_act_layer(from_layer, name, num_filter, kernel=(1,1), pad=(0,0), \
    stride=(1,1), act_type="relu", use_batchnorm=False):
    """
    wrapper for a small Convolution group

    Parameters:
    ----------
    from_layer : mx.symbol
        continue on which layer
    name : str
        base name of the new layers
    num_filter : int
        how many filters to use in Convolution layer
    kernel : tuple (int, int)
        kernel size (h, w)
    pad : tuple (int, int)
        padding size (h, w)
    stride : tuple (int, int)
        stride size (h, w)
    act_type : str
        activation type, can be relu...
    use_batchnorm : bool
        whether to use batch normalization

    Returns:
    ----------
    (conv, relu) mx.Symbols
    """
    assert not use_batchnorm, "batchnorm not yet supported"
    bias = mx.symbol.Variable(name="conv{}_bias".format(name),
        init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
    conv = mx.symbol.Convolution(data=from_layer, bias=bias, kernel=kernel, pad=pad, \
        stride=stride, num_filter=num_filter, name="conv{}".format(name))
    relu = mx.symbol.Activation(data=conv, act_type=act_type, \
        name="{}{}".format(act_type, name))
    if use_batchnorm:
        relu = mx.symbol.BatchNorm(data=relu, name="bn{}".format(name))
    return conv, relu

def  multi_layer_feature(body, from_layers, num_filters, strides, pads, min_filter=128):
    """Wrapper function to extract features from base network, attaching extra
    layers and SSD specific layers

    Parameters
    ----------
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    min_filter : int
        minimum number of filters used in 1x1 convolution

    Returns
    -------
    list of mx.Symbols

    """
    # arguments check
    assert len(from_layers) > 0
    assert isinstance(from_layers[0], str) and len(from_layers[0].strip()) > 0
    assert len(from_layers) == len(num_filters) == len(strides) == len(pads)

    # Lowest and highest pyramid levels in the backbone network. For FPN, we assume
    # that all networks have 5 spatial reductions, each by a factor of 2. Level 1
    # would correspond to the input image, hence it does not make sense to use it.
    LOWEST_BACKBONE_LVL = 2  # E.g., "conv2"-like level
    HIGHEST_BACKBONE_LVL = 5  # E.g., "conv5"-like level

    internals = body.get_internals()
    #out = internals.list_outputs()
    #print(out)
    backbone_layers = []
    for k, params in enumerate(zip(from_layers, num_filters, strides, pads)):
        from_layer, num_filter, s, p = params
        if from_layer.strip():
            # extract from base network
            layer = internals[from_layer.strip() + '_output']
            backbone_layers.append(layer)

    num_backbone_stages = len(backbone_layers)
    num_extra_stages = len(from_layers) - num_backbone_stages
    inner_blobs = []

    # For the coarsest backbone level: 1x1 conv only seeds recursion
    C5 = backbone_layers[-1]
    bias = mx.symbol.Variable(name="fpn_inner_stage5_conv_1x1_bias",
                              init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
    fpn_inner_stage5 = mx.symbol.Convolution(data=C5, kernel=(1, 1), pad=(0, 0), num_filter=256, bias=bias,
                                         name='fpn_inner_stage5_conv_1x1')
    inner_blobs.append(fpn_inner_stage5)

    #
    # Step 1: recursively build down starting from the coarsest backbone level
    #

    # For other levels add top-down and lateral connections
    for i in range(num_backbone_stages - 1):
        topdown = mx.symbol.UpSampling(inner_blobs[-1], scale=2, sample_type='nearest')
        bias = mx.symbol.Variable(name="fpn_inner_stage{}_conv_1x1_bias".format(4 - i),
                                  init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
        lateral = mx.symbol.Convolution(data=backbone_layers[num_backbone_stages - i - 2],
                                        kernel=(1, 1),
                                        pad=(0, 0),
                                        num_filter=256,
                                        bias=bias,
                                        name='fpn_inner_stage{}_conv_1x1'.format(4 - i))
        fpn_inner_stage = topdown + lateral
        inner_blobs.append(fpn_inner_stage)

    # Post-hoc scale-specific 3x3 convs
    blobs_fpn = []
    for i in range(num_backbone_stages):
        bias = mx.symbol.Variable(name="fpn_stage{}_conv_3x3_bias".format(5 - i),
                                  init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
        fpn_blob = mx.symbol.Convolution(data=inner_blobs[i],
                                         kernel=(3,3),
                                         pad=(1,1),
                                         num_filter=256,
                                         bias=bias,
                                         name='fpn_stage{}_conv_3x3'.format(5 - i))

        blobs_fpn += [fpn_blob]

    blobs_fpn.reverse()    # [P3, P4, P5]

    #
    # Step 2: build up starting from the coarsest backbone level
    #

    # Coarser FPN levels introduced for RetinaNet
    fpn_blob = C5
    for i in range(HIGHEST_BACKBONE_LVL + 1, HIGHEST_BACKBONE_LVL + num_extra_stages + 1):
        fpn_blob_in = fpn_blob
        if i > HIGHEST_BACKBONE_LVL + 1:
            fpn_blob_in = mx.symbol.Activation(fpn_blob, act_type='relu')
        bias = mx.symbol.Variable(name="fpn_stage{}_conv_3x3_bias".format(i),
                                  init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
        fpn_blob = mx.symbol.Convolution(fpn_blob_in,
                                         kernel=(3, 3),
                                         stride=(strides[i - 3], strides[i - 3]),
                                         pad=(pads[i - 3], pads[i - 3]),
                                         num_filter=num_filters[i - 3],
                                         bias=bias,
                                         name='fpn_stage{}_conv_3x3'.format(i))
        blobs_fpn.append(fpn_blob)

    return blobs_fpn    # [P3, P4, P5, P6, P7]

def multibox_layer(from_layers, num_classes, sizes=[.2, .95],
                    ratios=[1], normalization=-1, num_channels=[],
                    clip=False, interm_layer=0, steps=[]):
    """
    the basic aggregation module for SSD detection. Takes in multiple layers,
    generate multiple object detection targets by customized layers

    Parameters:
    ----------
    from_layers : list of mx.symbol
        generate multibox detection from layers
    num_classes : int
        number of classes excluding background, will automatically handle
        background in this function
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    num_channels : list of int
        number of input layer channels, used when normalization is enabled, the
        length of list should equals to number of normalization layers
    clip : bool
        whether to clip out-of-image boxes
    interm_layer : int
        if > 0, will add a intermediate Convolution layer
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions

    Returns:
    ----------
    list of outputs, as [loc_preds, cls_preds, anchor_boxes]
    loc_preds : localization regression prediction
    cls_preds : classification prediction
    anchor_boxes : generated anchor boxes
    """
    assert len(from_layers) > 0, "from_layers must not be empty list"
    assert num_classes > 0, \
        "num_classes {} must be larger than 0".format(num_classes)

    assert len(ratios) > 0, "aspect ratios must not be empty list"
    if not isinstance(ratios[0], list):
        # provided only one ratio list, broadcast to all from_layers
        ratios = [ratios] * len(from_layers)
    assert len(ratios) == len(from_layers), \
        "ratios and from_layers must have same length"

    assert len(sizes) > 0, "sizes must not be empty list"
    # if len(sizes) == 2 and not isinstance(sizes[0], list):
    #     # provided size range, we need to compute the sizes for each layer
    #      assert sizes[0] > 0 and sizes[0] < 1
    #      assert sizes[1] > 0 and sizes[1] < 1 and sizes[1] > sizes[0]
    #      tmp = np.linspace(sizes[0], sizes[1], num=(len(from_layers)-1))
    #      min_sizes = [start_offset] + tmp.tolist()
    #      max_sizes = tmp.tolist() + [tmp[-1]+start_offset]
    #      sizes = zip(min_sizes, max_sizes)
    assert len(sizes) == len(from_layers), \
        "sizes and from_layers must have same length"

    if not isinstance(normalization, list):
        normalization = [normalization] * len(from_layers)
    assert len(normalization) == len(from_layers)

    assert sum(x > 0 for x in normalization) <= len(num_channels), \
        "must provide number of channels for each normalized layer"

    if steps:
        assert len(steps) == len(from_layers), "provide steps for all layers or leave empty"

    loc_pred_layers = []
    cls_pred_layers = []
    anchor_layers = []
    num_classes += 1 # always use background as label 0

    # shared weights
    cls_conv1_weight = mx.symbol.Variable(name='cls_conv1_3x3_weight',
                                          init=mx.init.Normal(sigma=0.01))
    cls_conv1_bias = mx.symbol.Variable(name='cls_conv1_3x3_bias',
                                            init=mx.init.Constant(0.0), attr={'__lr_mult__': '1.0'})

    cls_conv2_weight = mx.symbol.Variable(name='cls_conv2_3x3_weight',
                                          init=mx.init.Normal(sigma=0.01))
    cls_conv2_bias = mx.symbol.Variable(name='cls_conv2_3x3_bias',
                                            init=mx.init.Constant(0.0), attr={'__lr_mult__': '1.0'})

    cls_conv3_weight = mx.symbol.Variable(name='cls_conv3_3x3_weight',
                                          init=mx.init.Normal(sigma=0.01))
    cls_conv3_bias = mx.symbol.Variable(name='cls_conv3_3x3_bias',
                                            init=mx.init.Constant(0.0), attr={'__lr_mult__': '1.0'})

    cls_conv4_weight = mx.symbol.Variable(name='cls_conv4_3x3_weight',
                                          init=mx.init.Normal(sigma=0.01))
    cls_conv4_bias = mx.symbol.Variable(name='cls_conv4_3x3_bias',
                                            init=mx.init.Constant(0.0), attr={'__lr_mult__': '1.0'})

    cls_score_weight = mx.symbol.Variable(name='cls_score_weight',
                                          init=mx.init.Normal(sigma=0.01))
    cls_score_bias = mx.symbol.Variable(name='cls_score_bias',
                                        init=mx.init.Constant(0.0), attr={'__lr_mult__': '1.0'})

    box_conv1_weight = mx.symbol.Variable(name='box_conv1_weight',
                                          init=mx.init.Normal(sigma=0.01))
    box_conv1_bias = mx.symbol.Variable(name='box_conv1_bias',
                                        init=mx.init.Constant(0.0), attr={'__lr_mult__': '1.0'})

    box_conv2_weight = mx.symbol.Variable(name='box_conv2_weight',
                                          init=mx.init.Normal(sigma=0.01))
    box_conv2_bias = mx.symbol.Variable(name='box_conv2_bias',
                                        init=mx.init.Constant(0.0), attr={'__lr_mult__': '1.0'})

    box_conv3_weight = mx.symbol.Variable(name='box_conv3_weight',
                                          init=mx.init.Normal(sigma=0.01))
    box_conv3_bias = mx.symbol.Variable(name='box_conv3_bias',
                                        init=mx.init.Constant(0.0), attr={'__lr_mult__': '1.0'})

    box_conv4_weight = mx.symbol.Variable(name='box_conv4_weight',
                                          init=mx.init.Normal(sigma=0.01))
    box_conv4_bias = mx.symbol.Variable(name='box_conv4_bias',
                                        init=mx.init.Constant(0.0), attr={'__lr_mult__': '1.0'})

    box_pred_weight = mx.symbol.Variable(name='box_pred_weight',
                                         init=mx.init.Normal(sigma=0.01))
    box_pred_bias = mx.symbol.Variable(name='box_pred_bias',
                                       init=mx.init.Constant(0.0), attr={'__lr_mult__': '1.0'})

    for k, from_layer in enumerate(from_layers):
        from_name = from_layer.name
        # normalize
        if normalization[k] > 0:
            from_layer = mx.symbol.L2Normalization(data=from_layer, \
                mode="channel", name="{}_norm".format(from_name))
            scale = mx.symbol.Variable(name="{}_scale".format(from_name),
                shape=(1, num_channels.pop(0), 1, 1),
                init=mx.init.Constant(normalization[k]),
                attr={'__wd_mult__': '0.1'})
            from_layer = mx.symbol.broadcast_mul(lhs=scale, rhs=from_layer)
        if interm_layer > 0:
            from_layer = mx.symbol.Convolution(data=from_layer, kernel=(3,3), \
                stride=(1,1), pad=(1,1), num_filter=interm_layer, \
                name="{}_inter_conv".format(from_name))
            from_layer = mx.symbol.Activation(data=from_layer, act_type="relu", \
                name="{}_inter_relu".format(from_name))

        # estimate number of anchors per location
        # here I follow the original version in caffe
        # TODO: better way to shape the anchors??
        size = sizes[k]
        assert len(size) > 0, "must provide at least one size"
        # size_str = "(" + ",".join([str(x) for x in size]) + ")"
        ratio = ratios[k]
        assert len(ratio) > 0, "must provide at least one ratio"
        # ratio_str = "(" + ",".join([str(x) for x in ratio]) + ")"
        num_anchors = len(size) * len(ratio)

        # create anchor generation layer
        if steps:
            step = (steps[k], steps[k])
        else:
            step = '(-1.0, -1.0)'

        anchors = []
        for r in ratio:
            anchor = mx.contrib.symbol.MultiBoxPrior(from_layer, sizes=size, ratios=r, \
                                                  clip=clip, steps=step)    # [1, h x w x 3, 4]
            anchor = mx.symbol.reshape(anchor, shape=(0, -1, len(size), 4))    # [1, h x w, 3, 4]
            anchors.append(anchor)

        anchors = mx.symbol.concat(*anchors, dim=2)    # [1, h x w, 9, 4]

        anchors = mx.symbol.Flatten(data=anchors)
        anchor_layers.append(anchors)

        # create location prediction layer
        num_loc_pred = num_anchors * 4
        box_conv1 = mx.symbol.Convolution(data=from_layer,
                                          weight=box_conv1_weight,
                                          bias=box_conv1_bias,
                                          kernel=(3, 3),
                                          stride=(1, 1),
                                          pad=(1, 1),
                                          num_filter=256,
                                          name="{}_loc_conv1_3x3".format(from_name))
        box_conv1 = mx.symbol.Activation(data=box_conv1, act_type='relu')
        box_conv2 = mx.symbol.Convolution(data=box_conv1,
                                          weight=box_conv2_weight,
                                          bias=box_conv2_bias,
                                          kernel=(3, 3),
                                          stride=(1, 1),
                                          pad=(1, 1),
                                          num_filter=256,
                                          name="{}_loc_conv2_3x3".format(from_name))
        box_conv2 = mx.symbol.Activation(data=box_conv2, act_type='relu')
        box_conv3 = mx.symbol.Convolution(data=box_conv2,
                                          weight=box_conv3_weight,
                                          bias=box_conv3_bias,
                                          kernel=(3, 3),
                                          stride=(1, 1),
                                          pad=(1, 1),
                                          num_filter=256,
                                          name="{}_loc_conv3_3x3".format(from_name))
        box_conv3 = mx.symbol.Activation(data=box_conv3, act_type='relu')
        box_conv4 = mx.symbol.Convolution(data=box_conv3,
                                          weight=box_conv4_weight,
                                          bias=box_conv4_bias,
                                          kernel=(3, 3),
                                          stride=(1, 1),
                                          pad=(1, 1),
                                          num_filter=256,
                                          name="{}_loc_conv4_3x3".format(from_name))
        box_conv4 = mx.symbol.Activation(data=box_conv4, act_type='relu')
        loc_pred = mx.symbol.Convolution(data=box_conv4,
                                         weight=box_pred_weight,
                                         bias=box_pred_bias,
                                         kernel=(3,3),
                                         stride=(1,1),
                                         pad=(1,1),
                                         num_filter=num_loc_pred,
                                         name="{}_loc_pred_conv".format(from_name))
        loc_pred = mx.symbol.transpose(loc_pred, axes=(0,2,3,1))
        loc_pred = mx.symbol.Flatten(data=loc_pred)
        loc_pred_layers.append(loc_pred)

        # create class prediction layer
        num_cls_pred = num_anchors * num_classes
        cls_conv1 = mx.symbol.Convolution(data=from_layer,
                                          weight=cls_conv1_weight,
                                          bias=cls_conv1_bias,
                                          kernel=(3, 3),
                                          stride=(1, 1),
                                          pad=(1, 1),
                                          num_filter=256,
                                          name="{}_cls_conv1_3x3".format(from_name))
        cls_conv1 = mx.symbol.Activation(data=cls_conv1, act_type='relu')
        cls_conv2 = mx.symbol.Convolution(data=cls_conv1,
                                          weight=cls_conv2_weight,
                                          bias=cls_conv2_bias,
                                          kernel=(3, 3),
                                          stride=(1, 1),
                                          pad=(1, 1),
                                          num_filter=256,
                                          name="{}_cls_conv2_3x3".format(from_name))
        cls_conv2 = mx.symbol.Activation(data=cls_conv2, act_type='relu')
        cls_conv3 = mx.symbol.Convolution(data=cls_conv2,
                                          weight=cls_conv3_weight,
                                          bias=cls_conv3_bias,
                                          kernel=(3, 3),
                                          stride=(1, 1),
                                          pad=(1, 1),
                                          num_filter=256,
                                          name="{}_cls_conv3_3x3".format(from_name))
        cls_conv3 = mx.symbol.Activation(data=cls_conv3, act_type='relu')
        cls_conv4 = mx.symbol.Convolution(data=cls_conv3,
                                          weight=cls_conv4_weight,
                                          bias=cls_conv4_bias,
                                          kernel=(3, 3),
                                          stride=(1, 1),
                                          pad=(1, 1),
                                          num_filter=256,
                                          name="{}_cls_conv4_3x3".format(from_name))
        cls_conv4 = mx.symbol.Activation(data=cls_conv4, act_type='relu')
        cls_pred = mx.symbol.Convolution(data=cls_conv4,
                                         weight=cls_score_weight,
                                         bias=cls_score_bias,
                                         kernel=(3,3),
                                         stride=(1,1),
                                         pad=(1,1),
                                         num_filter=num_cls_pred,
                                         name="{}_cls_pred_conv".format(from_name))
        cls_pred = mx.symbol.transpose(cls_pred, axes=(0,2,3,1))
        cls_pred = mx.symbol.Flatten(data=cls_pred)
        cls_pred_layers.append(cls_pred)

    loc_preds = mx.symbol.Concat(*loc_pred_layers, num_args=len(loc_pred_layers), \
        dim=1, name="multibox_loc_pred")
    cls_preds = mx.symbol.Concat(*cls_pred_layers, num_args=len(cls_pred_layers), \
        dim=1)
    cls_preds = mx.symbol.Reshape(data=cls_preds, shape=(0, -1, num_classes))
    cls_preds = mx.symbol.transpose(cls_preds, axes=(0, 2, 1), name="multibox_cls_pred")
    anchor_boxes = mx.symbol.Concat(*anchor_layers, \
        num_args=len(anchor_layers), dim=1)
    anchor_boxes = mx.symbol.Reshape(data=anchor_boxes, shape=(0, -1, 4), name="multibox_anchors")
    return [loc_preds, cls_preds, anchor_boxes]
