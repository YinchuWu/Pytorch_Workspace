import torch
import numpy as np



def write_results(prediction, confidencce, num_classes, nms_conf=0.4):
    conf_mask = (prediction[:, :, 4] > confidencce).float().unsqueeze(
        2)  # shape = N * GGA * 1
    prediction = prediction * conf_mask
    # print(prediction[:, :, 4])
    bbox_corner = prediction.new(prediction.shape)  # shape = N * GGA * attrs
    bbox_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    bbox_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    bbox_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    bbox_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    prediction[:, :, :4] = bbox_corner[:, :, :4]

    batch_size = bbox_corner.shape[0]
    write = False
    for ind in range(batch_size):
        image_pred = prediction[
            ind]  # image Tensor         # shape = GGA * attrs

        # confidence threshholding
        max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes],
                                             1)  # shape = GGA
        max_conf = max_conf.float().unsqueeze(1)  # shape = GGA * 1
        max_conf_score = max_conf_score.float().unsqueeze(1)  # shape = GGA * 1
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)  # shape = GGA * 5+2
        # print(image_pred.shape)
        non_zero_ind = torch.nonzero(
            image_pred[:, 4])  # shape = GAA[no_zero] * 1
        # For PyTorch 0.4 compatibility
        # Since the above code with not raise exception for no detection
        # as scalars are supported in PyTorch 0.4
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(
                -1, 7)  # shape = GAA[no_zero] * 7
        except:
            continue
        # handle situations with no valid_bbox
        if image_pred_.shape[0] == 0:
            continue
        # print(image_pred_[:, -1])
        # Get the various classes detected in the image
        img_classes = unique(  # shape : #detected_class
            image_pred_[:, -1])  # -1 index holds the class index
        # NMS
        # print(img_classes)
        for cls in img_classes:
            # perform NMS

            # get the detections with one particular class
            cls_mask = image_pred_ * (
                image_pred_[:, -1] == cls).float().unsqueeze(
                    1)  # shape = GAA[no_zero][detected_class] * 7
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = cls_mask[class_mask_ind, :].view(-1, 7)

            # sort the detections such that the entry with the maximum objectness
            # confidence is at the top
            # print(image_pred_class.shape)
            conf_sort_index = torch.sort(
                image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.shape[0]

            for i in range(idx):
                # Get the IOUs of all boxes that come after the one we are looking at in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0),
                                    image_pred_class[i + 1:])
                except ValueError:
                    break

                except IndexError:
                    break

                # Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i + 1:] *= iou_mask

                # Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)
        batch_ind = image_pred_class.new(image_pred_class.shape[0],
                                         1).fill_(ind)
        # Repeat the batch_id for as many detections of the class cls in the image
        seq = batch_ind, image_pred_class

        if not write:
            output = torch.cat(seq, 1)
            write = True
        else:
            out = torch.cat(seq, 1)
            output = torch.cat((output, out), 0)
    try:
        return output
    except:  # There isnt a single detection in any images of the batch
        return 0


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(
        inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(
            inter_rect_y2 - inter_rect_y1, min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area)

    return iou


def unique(tensor):
    s = set([])
    for i in tensor:
        k = i.item()
        s.add(k)
    s = np.array(list(s))
    return(torch.from_numpy(s).float().cuda())


if __name__ == '__main__':
    a = torch.from_numpy(np.arange(10))
    print(unique(a))
    # grid_size = 10
    # grid = np.arange(grid_size)
    # a, b = np.meshgrid(grid, grid)

    # x_offset = torch.FloatTensor(a).view(-1, 1)
    # y_offset = torch.FloatTensor(b).view(-1, 1)

    # x_y_offset = torch.cat((x_offset, y_offset),
    #                        1).repeat(1, 4).view(-1, 2).unsqueeze(0)
    # print(x_offset.shape)
    # print(x_y_offset.shape)

    # xx = torch.from_numpy(np.arange(8))
    # xx = xx.reshape((2, 2, 2))
    # y = xx[:, :, 1] > 6
    # y = y.float().unsqueeze(2)
    # t = y * xx.float()
    # z, c = torch.max(xx, 1)
    # # xx = xx.view(3, 3)
    # # c = xx.repeat(2, 2)
    # # z = z.unsqueeze(1)
    # a = torch.from_numpy(np.arange(9).reshape(3, 3))
    # b = torch.nonzero(a[:, 1])
    # # print(b)
    # print(a[b.squeeze(), :])
    # c = torch.from_numpy(np.arange(9).reshape((3, 3)))
    # d = torch.sort(c[:, 1])
    # e = torch.unique(c)
    # print(e)
    # # print(torch.unique(c[:,-1]))
