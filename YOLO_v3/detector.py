from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random

# python detector.py --images 'data/dog-cycle-car.png' --det 'det' --weights 'data/yolov3.weights'
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
        default=0.5)
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
        help=
        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
        default="416",
        type=str)

    return parser.parse_args()


def load_classes(namesfile):
    fp = open(namesfile, 'r')
    names = fp.read().split('\n')[:-1]
    return names


def load_network(args):
    #Set up the neural network
    print("Loading network.....")
    model = Darknet(args.cfgfile).cuda(device=None)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    #If there's a GPU availible, put the model on GPU
    if torch.cuda.is_available():
        model.cuda()

    #Set the model in evaluation mode
    model.eval()
    return model


def read_images(args,inp_dim):
    if not os.path.exists(args.det):
        os.makedirs(args.det)
    read_dir = time.time()
    #Detection phase
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
    #PyTorch Variables for images
    im_batches = list(
        map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))

    #List containing dimensions of original images
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
    return (imlist,im_batches)


def detection_loop(args, num_classes):
    write = 0
    start_det_loop = time.time()
    CUDA = torch.cuda.is_available()
    model = load_network(args)
    inp_dim = int(model.net_info["height"])
    imlist,im_batches = read_images(args,inp_dim)
    for i, batch in enumerate(im_batches):
        #load the image
        start = time.time()
        with torch.no_grad():
            prediction = model(Variable(batch).cuda(), CUDA)
            prediction = write_results(
                prediction,
                float(args.confidence),
                num_classes,
                nms_conf=float(args.nms_thresh))

        end = time.time()

        if type(prediction) == int:
            for im_num, image in enumerate(
                    imlist[i * int(agrs.bs):min((i + 1) *
                                                int(agrs.bs), len(imlist))]):
                im_id = i * int(agrs.bs) + im_num
                print("{0:20s} predicted in {1:6.3f} seconds".format(
                    image.split("/")[-1], (end - start) / int(agrs.bs)))
                print("{0:20s} {1:s}".format("Objects Detected:", ""))
                print(
                    "----------------------------------------------------------"
                )
            continue
        prediction[:,
                   0] += i * args.bs  #transform the atribute from index in batch to index in imlist

        if not write:                      #If we have't initialised output
            output = prediction
            write = 1
        else:
            output = torch.cat((output,prediction))

        for im_num, image in enumerate(imlist[i*args.bs: min((i +  1)*args.bs, len(imlist))]):
            im_id = i*args.bs + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/args.bs))
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            print("----------------------------------------------------------")

        if CUDA:
            torch.cuda.synchronize()
    try:
        output
    except NameError:
        print ("No detections were made")
        exit()



if __name__ == '__main__':
    args = arg_parse()
    # images = args.images
    # batch_size = int(args.bs)
    # confidence = float(args.confidence)
    # nms_thesh = float(args.nms_thresh)
    # start = 0


    num_classes = 80  #For COCO
    classes = load_classes("data/coco.names")
    detection_loop(args, num_classes)
