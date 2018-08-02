import torch.nn as nn


class DetectionLayer(nn.Module):
    """docstring for DetectionLayer"""

    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        # check the type of block
        # create a new module for the block
        # append to module_list

        if x['type'] == 'convolutional':
            # Get the info about the layer
            activation = x['activation']
            try:
                batch_normalized = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalized = 0
                bias = True
            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # Add the convolutional layer
            conv = nn.Conv2d(
                prev_filters,
                filters,
                kernel_size,
                stride=stride,
                padding=pad,
                dilation=1,
                groups=1,
                bias=bias)
            module.add_module('conv_{0}'.format(index), conv)

            # Add the Batch Norm Layer
            if batch_normalized:
                bn = nn.BatchNorm2d(
                    filters,
                    eps=1e-5,
                    momentum=0.1,
                    affine=True,
                    track_running_stats=True)
                module.add_module('batch_norm_{0}'.format(index), bn)

            # Check the activation.
            # It is either Linear or a Leaky ReLU for YOLO
            if activation == 'leaky':
                activn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
                module.add_module('Leaky_{0}'.format(index), activn)

        # If it's an upsampling layer
        # We use Bilinear2dUpsampling
        elif x['type'] == 'upsample':
            stride = int(x['stride'])
            upsample = nn.Upsample(
                size=None,
                scale_factor=2,
                mode='bilinear',
                align_corners=False)
            module.add_module('upsample_{0}'.format(index), upsample)

        # If it is a route layer
        elif x['type'] == 'route':
            x['layers'] = x['layers'].split(',')
            # Start  of a route
            start = int(x['layers'][0])
            # end, if there exists one.
            try:
                end = int(x['layers'][1])
            except:
                end = 0
            # Positive anotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index
                                                                         + end]
            else:
                filters = output_filters[index + start]

        # shortcut corresponds to skip connection
        elif x['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module('shortcut_{0}'.format(index), shortcut)

        # Yolo is the detection layer
        elif x['type'] == 'yolo':
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask]

            anchors = x['anchors'].split(',')
            anchors = [int(x) for x in anchors]
            anchors = [(anchors[i], anchors[i + 1])
                       for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module('detection_{0}'.format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)
