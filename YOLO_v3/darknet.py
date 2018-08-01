import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class Darknet(nn.Module):
    """docstring for Darknet"""

    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}  # We cache the outputs for the route layer

        write = 0
        for i, module in enumerate(modules):
            module_type = (module['type'])

            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.module_list[i](x)

            elif module_type == 'route':
                layers = module['layers']
                layers = [int(i) for i in layers]

                if layers[0] > 0:
                    layers[0] -= i

                if len(layers) == 1:
                    x = outputs[i + layers[0]]

                else:
                    if layers[1] > 0:
                        layers[1] -= i

                    map1 = outputs[layers[0]]
                    map2 = outputs[layers[1]]

                    x = torch.cat((map1, map2), 1)

            elif module_type == 'shortcut':
                from_ = module['from']
                x = outputs[i - 1] + outputs[i + from_]

            outputs[i] = x


class DetectionLayer(nn.Module):
    """docstring for DetectionLayer"""

    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """
    f = open(cfgfile, mode='r', buffering=-1, encoding=None,
             errors=None, newline=None, closefd=True)
    lines = f.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[':                         # This marks the start of a new block
            # If block is not empty, implies it is storing values of previous block.
            if len(block) != 0:
                blocks.append(block)               # add it the blocks list
                block = {}                         # re-init the block
            block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    return blocks


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
            conv = nn.Conv2d(prev_filters, filters, kernel_size,
                             stride=stride, padding=pad, dilation=1, groups=1, bias=bias)
            module.add_module('conv_{0}'.format(index), conv)

            # Add the Batch Norm Layer
            if batch_normalized:
                bn = nn.BatchNorm2d(
                    filters, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
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
            upsample = nn.UpsamplingBilinear2d(size=None, scale_factor=2)
            module.add_module('upsample_{0}'.format(index), upsample)

        # If it is a route layer
        elif x['type'] == 'route':
            x['layers'] = x['layers'].split(',')
            # Start  of a route
            start = int(layers[0])
            # end, if there exists one.
            try:
                end = int(layers[1])
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
                filters = output_filters[index +
                                         start] + output_filters[index + end]
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


if __name__ == '__main__':
    darknet = Darknet('cfg/yolov3.cfg')
    print(darknet.module_list[0])
