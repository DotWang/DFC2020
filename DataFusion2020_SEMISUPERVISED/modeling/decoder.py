import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        #x = F.upsample_bilinear(x, size=low_level_feat.size()[2:])
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class FPNBlock(nn.Module):

    def __init__(self, skip_channels, pyramid_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x):
        x, skip = x

        x = F.interpolate(x, scale_factor=2, mode='bilinear',align_corners=True)
        skip = self.skip_conv(skip)
        if skip.shape!=x.shape:
            skip=F.interpolate(skip, size=x.size()[2:], mode='bilinear',align_corners=True)

        x = x + skip
        return x

class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [
            Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))
        ]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)

class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):

        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3),
                              stride=1, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x

class deeplabFPNDecoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, pyramid_channels=256,segmentation_channels=128,):
        super(deeplabFPNDecoder, self).__init__()
        if backbone == 'resnet':
            encoder_channels=[64,256,512]
            end_inplanes=[1024,2048]
        else:
            raise NotImplementedError

        self.conv0=nn.Conv2d(end_inplanes[0], pyramid_channels, kernel_size=1, stride=1)
        self.conv1=nn.Conv2d(end_inplanes[1], pyramid_channels, kernel_size=1, stride=1)
        self.p3 = FPNBlock(encoder_channels[-1],pyramid_channels)
        self.p2 = FPNBlock(encoder_channels[-2], pyramid_channels)
        self.conv2=nn.Conv2d(encoder_channels[-3], pyramid_channels, kernel_size=1, stride=1)

        self.s4 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=2)
        self.s3 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=1)
        self.s2 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=0)
        self.s1 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=0)

        self.conv = nn.Sequential(nn.Conv2d(256+4*segmentation_channels, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU())
        self.last_conv = nn.Sequential(nn.Conv2d(256+encoder_channels[0], 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()



    def forward(self, xa ,x, x0, x1 ,x2, x3, x4):

        p4 = self.conv0(x4)+self.conv1(x)
        p3 = self.p3([p4,x3])
        p2 = self.p2([p3,x2])#256,64,64
        p1 = self.conv2(x1)+p2

        s4 = self.s4(p4)
        s3 = self.s3(p3)
        s2 = self.s2(p2)
        s1 = self.s1(p1)

        features=torch.cat((s4,s3,s2,s1),dim=1)

        #x = F.upsample_bilinear(x, size=low_level_feat.size()[2:])
        x = F.interpolate(xa, size=features.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, features), dim=1)
        x = self.conv(x)
        x = F.interpolate(x, size=x0.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, x0), dim=1)
        x = self.last_conv(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)

class spatial_attention(nn.Module):

    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()

        assert kernel_size in (3, 7),'kernel size must be 3 or 7'

        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1,kernel_size=kernel_size, padding=padding)

    def forward(self,x):
        avg_x=torch.mean(x, dim=1, keepdim=True)
        max_x,_=torch.max(x, dim=1, keepdim=True)
        x=torch.cat((avg_x,max_x),1)
        x=self.conv(x)
        #print('spatial',x.size())
        x=torch.sigmoid(x)
        return x

class channel_attention(nn.Module):
    def __init__(self, out_channels):
        super(channel_attention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv2d(out_channels, int(out_channels/16),kernel_size=1, padding=0, bias=False)
        self.relu=nn.ReLU()
        self.conv2 = nn.Conv2d(int(out_channels/16), out_channels,kernel_size=1, padding=0, bias=False)

    def forward(self,x):
        avg_x = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
        max_x = self.conv2(self.relu(self.conv1(self.max_pool(x))))
        x = avg_x + max_x
        x=torch.sigmoid(x)
        return x

class UnetDecoder(nn.Module):

    def __init__(self, in_channels, channels, out_channels):
        super(UnetDecoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, out_channels,kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.spatial_gate = spatial_attention()
        self.channel_gate = channel_attention(out_channels)

    def forward(self, x, e=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        #print('x',x.size())
        #print('e',e.size())
        if e is not None:
            x = torch.cat([x,e],1)

        x = F.relu(self.conv1(x),inplace=True)
        x1 = F.relu(self.conv2(x),inplace=True)

        g1 = self.channel_gate(x1)
        x2 = g1*x1
        g2 = self.spatial_gate(x2)
        x = g2*x2
        x = x + x1
        return g1,g2,x