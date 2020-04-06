from modeling.backbone.resnet_original import resnet50,resnet101,resnext50_32x4d
from modeling.backbone.seresnet import se_resnet_50
from modeling.backbone.seresnext import se_resnext_50
from modeling.backbone.cbamnet import cbam_resnet50
from modeling.backbone.cbamnext import cbam_resnext_50

def unet_encoder(backbone,depth,in_channel,pretrn):
    if backbone=='resnet':
        if depth==50:
            print("****************backbone is ResNet50****************")
            return resnet50(in_channel, pretrained=pretrn)
        elif depth==101:
            print("****************backbone is ResNet101****************")
            return resnet101(in_channel, pretrained=pretrn)
        else:
            raise NotImplementedError
    elif backbone=='resnext':
        print("****************backbone is ResNeXt50****************")
        return resnext50_32x4d(in_channel, pretrained=pretrn)

    elif backbone=='senet':
        print("****************backbone is SE-ResNet50****************")
        return se_resnet_50(in_channel,pretrained=pretrn)

    elif backbone=='senext':
        print("****************backbone is SE-ResNeXt50****************")
        return se_resnext_50(in_channel)

    elif backbone=='cbamnet':
        print("****************backbone is CBAM-ResNet50****************")
        return cbam_resnet50(in_channel,pretrained=pretrn)
    elif backbone=='cbamnext':
        print("****************backbone is CBAM-ResNeXt50****************")
        return cbam_resnext_50(in_channel)





