import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder,deeplabFPNDecoder,UnetDecoder
from modeling.backbone import build_backbone
from modeling.backbone.resnet_original import resnet50,resnet34,resnet101
#from torch_receptive_field import receptive_field


class DeepLab(nn.Module):
    def __init__(self, args, backbone='resnet', output_stride=16, num_classes=4,
                 sync_bn=False, freeze_bn=False, depth=50):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8
        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.args = args
        if self.args.oly_s1 and not self.args.oly_s2:
            in_channel = 2
            pretrn=False
        elif not self.args.oly_s1 and self.args.oly_s2 and not self.args.rgb:
            in_channel = 10
            pretrn = False
        elif not self.args.oly_s1 and self.args.oly_s2 and self.args.rgb:
            in_channel = 3
            pretrn = True
        elif not self.args.oly_s1 and not self.args.oly_s2 and not self.args.rgb:
            in_channel = 12
            pretrn = False
        elif not self.args.oly_s1 and not self.args.oly_s2 and self.args.rgb:
            in_channel = 5
            pretrn = False
        else:
            raise NotImplementedError
        self.backbone = build_backbone(backbone, in_channel, output_stride, BatchNorm, depth, pretrn)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()
        self.freeze_backbone()

    def forward(self, s1,s2):
        if self.args.oly_s1 and not self.args.oly_s2:
            input = s1
        elif not self.args.oly_s1 and self.args.oly_s2:
            input = s2
        elif not self.args.oly_s1 and not self.args.oly_s1:
            input = torch.cat((s1, s2), 1)
        else:
            raise NotImplementedError
        x,_, _, low_level_feat, _, _ = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        output = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x,output

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def freeze_backbone(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():                
                for p in m[1].parameters():
                    p.requires_grad = False

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

class DeeplabFPN(nn.Module):

    def __init__(self, args, backbone='resnet', output_stride=16, num_classes=4,
                 sync_bn=False, freeze_bn=False, depth=50):
        super(DeeplabFPN, self).__init__()

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.args=args

        if self.args.oly_s1 and not self.args.oly_s2:
            in_channel = 2
            pretrn=False
        elif not self.args.oly_s1 and self.args.oly_s2 and not self.args.rgb:
            in_channel = 10
            pretrn = False
        elif not self.args.oly_s1 and self.args.oly_s2 and self.args.rgb:
            in_channel = 3
            pretrn = True
        elif not self.args.oly_s1 and not self.args.oly_s2 and not self.args.rgb:
            in_channel = 12
            pretrn = False
        elif not self.args.oly_s1 and not self.args.oly_s2 and self.args.rgb:
            in_channel = 5
            pretrn = False
        else:
            raise NotImplementedError

        self.backbone = build_backbone(backbone,in_channel,output_stride, BatchNorm, depth, pretrn)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = deeplabFPNDecoder(num_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()
        self.freeze_backbone()

    def forward(self, s1,s2):
        if self.args.oly_s1 and not self.args.oly_s2:
            input = s1
        elif not self.args.oly_s1 and self.args.oly_s2:
            input = s2
        elif not self.args.oly_s1 and not self.args.oly_s1:
            input = torch.cat((s1, s2), 1)
        else:
            raise NotImplementedError

        x,x0,x1,x2,x3,x4 = self.backbone(input)
        xa = self.aspp(x)
        x = self.decoder(xa,x,x0,x1,x2,x3,x4)
        output = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x,output

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def freeze_backbone(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                for p in m[1].parameters():
                    p.requires_grad = False

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1],nn.GroupNorm):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

class Unet(nn.Module):
    def __init__(self, args,num_classes=10,depth=34):
        super().__init__()

        self.args=args

        if self.args.oly_s1 and not self.args.oly_s2:
            in_channel = 2
            pretrn=False
        elif not self.args.oly_s1 and self.args.oly_s2 and not self.args.rgb:
            in_channel = 10
            pretrn = False
        elif not self.args.oly_s1 and self.args.oly_s2 and self.args.rgb:
            in_channel = 3
            pretrn = True
        elif not self.args.oly_s1 and not self.args.oly_s2 and not self.args.rgb:
            in_channel = 12
            pretrn = False
        elif not self.args.oly_s1 and not self.args.oly_s2 and self.args.rgb:
            in_channel = 5
            pretrn = False
        else:
            raise NotImplementedError

        if depth==34:
            self.backbone = resnet34(in_channel, pretrained=pretrn)

            self.center = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

            self.decoder5 = UnetDecoder(256 + 512, 512, 64)
            self.decoder4 = UnetDecoder(64 + 256, 256, 64)
            self.decoder3 = UnetDecoder(64 + 128, 128, 64)
            self.decoder2 = UnetDecoder(64 + 64, 64, 64)
            self.decoder1 = UnetDecoder(64 + in_channel, 32, 64)

        if depth==50:
            self.backbone = resnet50(in_channel, pretrained=pretrn)

            self.center = nn.Sequential(
                nn.Conv2d(2048, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

            self.decoder5 = UnetDecoder(256 + 2048, 512, 64)
            self.decoder4 = UnetDecoder(64 + 1024, 256, 64)
            self.decoder3 = UnetDecoder(64 + 512, 128, 64)
            self.decoder2 = UnetDecoder(64 + 256, 64, 64)
            self.decoder1 = UnetDecoder(64, 32, 64)

            self.logit=nn.Sequential(
                nn.Conv2d(64,64,kernel_size=1,padding=0),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.5),
                nn.Conv2d(64, num_classes, kernel_size=1, padding=0)
            )

        if depth==101:
            self.backbone = resnet101(in_channel, pretrained=pretrn)

            self.center = nn.Sequential(
                nn.Conv2d(2048, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

            self.decoder5 = UnetDecoder(256 + 2048, 512, 64)
            self.decoder4 = UnetDecoder(64 + 1024, 256, 64)
            self.decoder3 = UnetDecoder(64 + 512, 128, 64)
            self.decoder2 = UnetDecoder(64 + 256, 64, 64)
            self.decoder1 = UnetDecoder(64, 32, 64)

            self.logit=nn.Sequential(
                nn.Conv2d(64,64,kernel_size=1,padding=0),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.5),
                nn.Conv2d(64, num_classes, kernel_size=1, padding=0)
            )


        self.freeze_backbone()

    def forward(self, s1,s2):

        if self.args.oly_s1 and not self.args.oly_s2:
            input = s1
        elif not self.args.oly_s1 and self.args.oly_s2:
            input = s2
        elif not self.args.oly_s1 and not self.args.oly_s1:
            input = torch.cat((s1, s2), 1)
        else:
            raise NotImplementedError


        e1,e2,e3,e4, e5=self.backbone(input)

        #print('e5', e5.shape)
        # print('e1', e2.shape)
        # print('e1', e3.shape)
        # print('e1', e4.shape)
        # print('e1', e5.shape)

        f = self.center(e5)

        d5 = self.decoder5(f, e5)
        #print('d5', d5.shape)
        d4 = self.decoder4(d5, e4)
        #print('d4', d4.shape)
        d3 = self.decoder3(d4, e3)
        #print('d3', d3.shape)
        d2 = self.decoder2(d3, e2)
        #print('d2', d2.shape)
        d1 = self.decoder1(d2)


        output=self.logit(d1)

        output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)

       # print(output.shape)

        return output

    def freeze_backbone(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                for p in m[1].parameters():
                    p.requires_grad = False

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.center, self.decoder5,self.decoder4,self.decoder3,self.decoder2,self.decoder1,self.logit]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.GroupNorm):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test")
    args = parser.parse_args()
    args.oly_s1=False
    args.rgb=True
    args.oly_s2=True
    model = Unet(args,depth=101)
    model.eval()
    #os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    # model = DeepLab(backbone='resnet', output_stride=16).to(device)
    # receptive_field(model, input_size=(3, 256, 256))
    input1 = torch.rand(1, 1, 256, 256)
    input2 = torch.rand(1, 3, 256, 256)
    output = model(input1,input2)
    print(output.shape)


