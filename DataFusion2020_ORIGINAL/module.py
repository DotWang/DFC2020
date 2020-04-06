import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from scipy.ndimage import zoom
from utils.func import CRF
import CC_labeling_8
from skimage import measure
from lovasz_losses import lovasz_softmax
import torch.nn.functional as F

min_prob = 0.0001

def CELossLayer(args,output,y):

    y = torch.squeeze(y, 1)
    y[y == 2] = 10
    y[y == 7] = 10
    #y[y == 9] = 10

    y = y.cuda()


    criterion = nn.CrossEntropyLoss(ignore_index=10, size_average=True)

    criterion = criterion.cuda()

    loss=criterion(output,y.long())

    return loss


def FocalLossLayer(args, output, y, gamma=2, alpha=0.5):

    # x: N,C,H,W
    # cues: N,C+1,H,W

    # cues = cues.argmax(1)
    # cues[cues == 2] = 10
    # cues[cues == 7] = 10

    y = torch.squeeze(y, 1)
    y[y == 2] = 10
    y[y == 7] = 10
    y[y == 9] = 10

    y = y.cuda()

    if not args.weighted:
        val_weight = torch.Tensor([1, 1, 0, 1, 1, 1, 1, 0, 1])

    elif args.weighted:
        val_weight = torch.Tensor([1.5789, 8.2345, 0, 1.2960, 14.5416, 0.7881, 1.3228, 0, 83.5003, 0.2289])

    criterion = nn.CrossEntropyLoss(weight=val_weight, ignore_index=10, size_average=True)

    criterion = criterion.cuda()

    logpt = -criterion(output, y.long())

    pt = torch.exp(logpt)

    if alpha is not None:
        logpt *= alpha

    loss = -((1 - pt) ** gamma) * logpt
    #print('focal loss',loss)

    return loss

def LovaszLossLayer(output, y):

    # x: N,C,H,W
    # cues: N,C+1,H,W
    # y:N,1,H,W

    # cues = cues.argmax(1)
    # cues[cues == 2] = 10
    # cues[cues == 7] = 10
    # loss1=lovasz_softmax(x, cues, classes='present', per_image=False, ignore=10)

    y = torch.squeeze(y, 1)
    y[y == 30] = 10
    y[y == 31] = 10
    y[y == 7] = 10
    #y[y == 9] = 10

    y = y.cuda()

    loss = lovasz_softmax(output, y, classes='present', per_image=False, ignore=10)

    return loss
    #
    # print('cues lovasz',loss1)
    # print('labels lovasz',loss2)









