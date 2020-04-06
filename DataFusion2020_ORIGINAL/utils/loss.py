import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=False, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode=='soft':
            return self.SoftCrossEntropy
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def SoftCrossEntropy(self, logit, target):
        n, c, h, w = logit.size()

        label=target.deepcopy()

        label[label==255]=4

        tmp=torch.zeros(n,c+1,h,w).cuda()

        for i in range(n):
            tmp[i]=tmp[i].scatter_(0,label[[i]].long(),1)

        label=tmp[:,:-1,:,:]#N,C,H,W

        label=np.array(label.cpu())
        for i in range(n):
            for j in range(c):
                label[i,j,:,:]=cv2.GaussianBlur(label[i,j,:,:],(11,11),0)

        label=torch.Tensor(label).cuda()#N,C,H,W

        log_likelihood = -F.log_softmax(logit, dim=1)

        loss = torch.sum(torch.mul(log_likelihood, label))/(h*w*c)

        return loss


    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




