from modeling.backbone import resnet, xception, drn, mobilenet, vgg, densenet, inception, squeezenet


def build_backbone(backbone,in_channel,output_stride, BatchNorm, depth, pretrn):
    if backbone == 'resnet':
        if depth == 50:
            print("****************backbone is ResNet50****************")
            return resnet.ResNet50(in_channel,output_stride, BatchNorm, pretrained=pretrn)
        elif depth == 101:
            print("****************backbone is ResNet101****************")
            return resnet.ResNet101(output_stride, BatchNorm)
        elif depth == 152:
            print("****************backbone is ResNet152****************")
            return resnet.ResNet152(output_stride, BatchNorm)
        else:
            print("there no match depth resnet backbone!")
    elif backbone == 'xception':
        print("****************backbone is xception****************")
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        print("****************backbone is drn_d_54****************")
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        print("****************backbone is MobileNetV2****************")
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    elif backbone == "vgg":
        if depth == 16:
            print("****************backbone is vgg16****************")
            return vgg.vgg16(output_stride, BatchNorm)
        elif depth == 17:
            print("****************backbone is vgg16_bn****************")
            return vgg.vgg16_bn(output_stride, BatchNorm)
        elif depth == 19:
            print("****************backbone is vgg19****************")
            return vgg.vgg19(output_stride, BatchNorm)
        elif depth == 20:
            print("****************backbone is vgg19_bn****************")
            return vgg.vgg19_bn(output_stride, BatchNorm)
        else:
            print("there no match depth vgg backbone!")
    elif backbone == "densenet":
        if depth == 121:
            print("****************backbone is densenet121****************")
            return densenet.densenet121(output_stride, BatchNorm)
        elif depth == 169:
            print("****************backbone is densenet136****************")
            return densenet.densenet136(output_stride, BatchNorm)
        elif depth == 201:
            print("****************backbone is densenet201****************")
            return densenet.densenet201(output_stride, BatchNorm)
        elif depth == 161:
            print("****************backbone is densenet161****************")
            return densenet.densenet161(output_stride, BatchNorm)
        else:
            print("there no match depth densenet backbone!")
    elif backbone == "inception":
        print("****************backbone is inception_v3_google****************")
        return inception.inception_v3_google
    elif backbone == "squeezenet":
        if depth == 1:
            print("****************backbone is squeezenet1_0****************")
            return squeezenet.squeezenet1_0(output_stride, BatchNorm)
        elif depth == 2:
            print("****************backbone is squeezenet1_1****************")
            return squeezenet.squeezenet1_1(output_stride, BatchNorm)
        else:
            print("there no match depth squeezenet backbone!")
    else:
        raise NotImplementedError
