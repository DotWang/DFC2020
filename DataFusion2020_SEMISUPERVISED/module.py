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

class WeightEMA(object):
    def __init__(self, params, src_params, alpha):
        self.params = list(params)
        self.src_params = list(src_params)
        self.alpha = alpha

        for p, src_p in zip(self.params, self.src_params):
            p.data[:] = src_p.data[:]

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for p, src_p in zip(self.params, self.src_params):
            p.data.mul_(self.alpha)
            p.data.add_(src_p.data * one_minus_alpha)

def CELossLayer(output,y):

    y = torch.squeeze(y, 1)

    # y[y == 30] = 10
    # y[y == 31] = 10
    # y[y == 2] = 10
    # y[y == 7] = 10

    y = y.cuda()


    criterion = nn.CrossEntropyLoss(ignore_index=10, size_average=True)

    criterion = criterion.cuda()

    loss=criterion(output,y.long())

    return loss


def FocalLossLayer(output, y, gamma=2, alpha=0.5):

    # x: N,C,H,W
    # cues: N,C+1,H,W

    # cues = cues.argmax(1)
    # cues[cues == 2] = 10
    # cues[cues == 7] = 10

    y = torch.squeeze(y, 1)
    # y[y == 30] = 10
    # y[y == 31] = 10
    # y[y == 2] = 10
    # y[y == 7] = 10

    y = y.cuda()

    criterion = nn.CrossEntropyLoss(ignore_index=10, size_average=True)

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
    # y[y == 30] = 10
    # y[y == 31] = 10
    # y[y == 2] = 10
    # y[y == 7] = 10
    # #y[y == 9] = 10

    y = y.cuda()

    loss = lovasz_softmax(output, y, classes='present', per_image=False, ignore=10)

    return loss
    #
    # print('cues lovasz',loss1)
    # print('labels lovasz',loss2)

def ConsistencyLossLayer(x1,x2):

    loss = nn.MSELoss()(x1,x2)

    return loss







