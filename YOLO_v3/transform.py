import torch
import numpy as np
import cv2


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """
    img = cv2.resize(img, (inp_dim, inp_dim))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img


def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(
        img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
    canvas[(h - new_h) // 2:(h - new_h) // 2 +
           new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image
    return canvas


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):

    # prediction : shape: N * C * G * G
    # anchor : shape: A * 2
    # C = A * attrs
    # attrs = [x,y,w,h,scores,class]
    batch_size = prediction.size(0)  # = N
    stride = inp_dim // prediction.size(2)  # = I/G
    grid_size = inp_dim // stride  # = G
    bbox_attrs = 5 + num_classes  # = attrs
    num_anchors = len(anchors)  # = A

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors,
                                 grid_size * grid_size)  # shape = N * C * GG
    prediction = prediction.transpose(1, 2).contiguous()  # shape = N * GG * C
    prediction = prediction.view(  # shape = N * GGA * attrs
        batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

    anchors = [(a[0] / stride, a[1] / stride)
               for a in anchors]  # shape = A * 2

    # Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # Add the center offsets
    grid = np.arange(grid_size)  # shape = G * 1
    a, b = np.meshgrid(grid, grid)  # shape = G * G

    x_offset = torch.FloatTensor(a).view(-1, 1)  # shape = GG * 1
    y_offset = torch.FloatTensor(b).view(-1, 1)  # shape = GG * 1
    
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    x_y_offset = torch.cat(
        (x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(
            0)  # shape = 1 * GGA * 2

    prediction[:, :, :2] += x_y_offset  # shape N * G*G*A * 2

    # log space transform height and the width
    anchors = torch.FloatTensor(anchors)  # shape A * 2

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size,
                             1).unsqueeze(0)  # shape 1 * G*G*A * 2
    prediction[:, :, 2:4] = torch.exp(
        prediction[:, :, 2:4]) * anchors  # shape N * GGA * 2
    prediction[:, :, 5:5 + num_classes] = torch.sigmoid(
        (prediction[:, :, 5:5 + num_classes]))

    prediction[:, :, :4] *= stride

    return prediction  # N * GGA * attrs

if __name__ == '__main__':
    x = torch.Tensor(np.arange(10)).cuda().view(-1,1)
    y = np.arange(10)
    a,b = np.meshgrid(y,y)
    a,b = torch.FloatTensor(a).view(-1,1),torch.FloatTensor(b).view(-1,1)
    c = torch.cat((a,b),1).repeat(1,2)
    print(c.shape)