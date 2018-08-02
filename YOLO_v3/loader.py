import torch
from darknet import Darknet
import os
import os.path as osp
from torch.autograd import Variable
import cv2
import time
from transform import prep_image
import numpy as np


def load_classes(namesfile):
    fp = open(namesfile, 'r')
    names = fp.read().split('\n')[:-1]
    return names


def load_network(args):
    # Set up the neural network
    print("Loading network.....")
    model = Darknet(args.cfgfile).cuda(device=None)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # If there's a GPU availible, put the model on GPU
    if torch.cuda.is_available():
        model.cuda()

    # Set the model in evaluation mode
    model.eval()
    return model


def load_images(args, inp_dim):
    if not os.path.exists(args.det):
        os.makedirs(args.det)
    read_dir = time.time()
    # Detection phase
    try:
        imlist = [
            osp.join(osp.realpath('.'), args.images, img)
            for img in os.listdir(args.images)
        ]
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), args.images))
    except FileNotFoundError:
        print("No file or directory with the name {}".format(args.images))
        exit()

    load_batch = time.time()
    loaded_ims = [cv2.imread(x) for x in imlist]
    # PyTorch Variables for images
    im_batches = list(
        map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))

    # List containing dimensions of original images
    im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

    if torch.cuda.is_available():
        im_dim_list = im_dim_list.cuda()
    batch_size = int(args.bs)
    leftover = 0
    if (len(im_dim_list) % batch_size):
        leftover = 1

    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover
        im_batches = [
            torch.cat(
                (im_batches[i * batch_size:min((i + 1) *
                                               batch_size, len(im_batches))]))
            for i in range(num_batches)
        ]
    return (imlist, im_batches, im_dim_list, loaded_ims, read_dir, load_batch)


def load_test_input(img_file):
    img = cv2.imread(img_file)
    img = cv2.resize(img, (416, 416))  # Resize to the input dimension
    # BGR -> RGB | H X W C -> C X H X W
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    # Add a channel at 0 (for batch) | Normalise
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()  # Convert to float
    img_ = Variable(img_)  # Convert to Variable
    return img_


if __name__ == '__main__':
    model = Darknet("cfg/yolov3.cfg").cuda(device=None)
    model.load_weights("data/yolov3.weights")
    inp = load_test_input('data/dog-cycle-car.png').cuda()
    pred = model(inp, torch.cuda.is_available())
    result = write_results(
        pred.data,
        0.0,
        80,
    )
    print(result.shape)