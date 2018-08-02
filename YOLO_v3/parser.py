import argparse


def arg_parse():
    """
    Parse arguements to the detect module
    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument(
        "--images",
        dest='images',
        help="Image / Directory containing images to perform detection upon",
        default="imgs",
        type=str)
    parser.add_argument(
        "--det",
        dest='det',
        help="Image / Directory to store detections to",
        default="det",
        type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument(
        "--confidence",
        dest="confidence",
        help="Object Confidence to filter predictions",
        default=0.2)
    parser.add_argument(
        "--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument(
        "--cfg",
        dest='cfgfile',
        help="Config file",
        default="cfg/yolov3.cfg",
        type=str)
    parser.add_argument(
        "--weights",
        dest='weightsfile',
        help="weightsfile",
        default="yolov3.weights",
        type=str)
    parser.add_argument(
        "--reso",
        dest='reso',
        help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
        default="416",
        type=str)

    return parser.parse_args()


def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """
    f = open(
        cfgfile,
        mode='r',
        buffering=-1,
        encoding=None,
        errors=None,
        newline=None,
        closefd=True)
    lines = f.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[':  # This marks the start of a new block
            # If block is not empty, implies it is storing values of previous block.
            if len(block) != 0:
                blocks.append(block)  # add it the blocks list
                block = {}  # re-init the block
            block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    return blocks
