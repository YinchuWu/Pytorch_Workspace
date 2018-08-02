from __future__ import division
import time
import torch
from torch.autograd import Variable
import cv2
from util import write_results
import pickle as pkl
import pandas as pd
from parser import arg_parse
from loader import load_network, load_images, load_classes


# python detector.py --images 'data/dog-cycle-car.png' --det 'det' --weights 'data/yolov3.weights'
def detection_loop(args, num_classes):
    write = 0
    start_det_loop = time.time()
    CUDA = torch.cuda.is_available()
    model = load_network(args)
    inp_dim = int(model.net_info["height"])
    imlist, im_batches, im_dim_list, loaded_ims, read_dir, load_batch = load_images(
        args, inp_dim)
    for i, batch in enumerate(im_batches):
        # load the image
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
                    imlist[i * int(args.bs):min((i + 1) *
                                                int(args.bs), len(imlist))]):
                im_id = i * int(args.bs) + im_num
                print("{0:20s} predicted in {1:6.3f} seconds".format(
                    image.split("/")[-1], (end - start) / int(args.bs)))
                print("{0:20s} {1:s}".format("Objects Detected:", ""))
                print(
                    "----------------------------------------------------------"
                )
            continue
        prediction[:,
                   0] += i * args.bs  # transform the atribute from index in batch to index in imlist

        if not write:  # If we have't initialised output
            output = prediction
            write = 1
        else:
            output = torch.cat((output, prediction))

        for im_num, image in enumerate(
                imlist[i * args.bs:min((i + 1) * args.bs, len(imlist))]):
            im_id = i * args.bs + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            print("{0:20s} predicted in {1:6.3f} seconds".format(
                image.split("/")[-1], (end - start) / args.bs))
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            print("----------------------------------------------------------")

        if CUDA:
            torch.cuda.synchronize()
    try:
        output
    except NameError:
        print("No detections were made")
        exit()

    im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

    scaling_factor = torch.min(inp_dim / im_dim_list, 1)[0].view(-1, 1)

    output[:, [1, 3]] -= (
        inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
    output[:, [2, 4]] -= (
        inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

    output[:, 1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0,
                                        im_dim_list[i, 0])
        output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0,
                                        im_dim_list[i, 1])

    class_load = time.time()
    colors = pkl.load(open("data/pallete", "rb"))
    draw = time.time()

    def Write(x, results, color):
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        img = results[int(x[0])]
        cls = int(x[-1])
        label = "{0}".format(classes[cls])
        cv2.rectangle(img, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4),
                    cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
        return img

    list(map(lambda x: Write(x, loaded_ims, colors[1]), output))
    det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det, x.split("/")[-1]))

    list(map(cv2.imwrite, det_names, loaded_ims))
    end = time.time()

    print("SUMMARY")
    print("----------------------------------------------------------")
    print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
    print()
    print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
    print("{:25s}: {:2.3f}".format("Loading batch",
                                   start_det_loop - load_batch))
    print("{:25s}: {:2.3f}".format(
        "Detection (" + str(len(imlist)) + " images)",
        output_recast - start_det_loop))
    print("{:25s}: {:2.3f}".format("Output Processing",
                                   class_load - output_recast))
    print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
    print("{:25s}: {:2.3f}".format("Average time_per_img",
                                   (end - load_batch) / len(imlist)))
    print("----------------------------------------------------------")

    torch.cuda.empty_cache()


if __name__ == '__main__':
    args = arg_parse()
    # images = args.images
    # batch_size = int(args.bs)
    # confidence = float(args.confidence)
    # nms_thesh = float(args.nms_thresh)
    # start = 0

    num_classes = 80  # For COCO
    classes = load_classes("data/coco.names")
    detection_loop(args, num_classes)
