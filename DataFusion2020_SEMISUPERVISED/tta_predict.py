import argparse
import os
import numpy as np
import torch
import rasterio
from tqdm import tqdm
from libtiff import TIFF
from scipy.misc import imsave
from PIL import Image
from torch.utils.data import DataLoader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.networks import *
from DFC2020 import DF2020
from utils.func import *
from utils.metrics import Evaluator
from val_tes_dataloader import load_tesdata,load_valdata



def DrawResult(labels, row, col):
    num_class = 10

    X_result = np.zeros((labels.shape[0], 3))
    for i in range(num_class):
        X_result[np.where(labels == i), 0] = Hex_to_RGB(hex_color_dict[i])[0]
        X_result[np.where(labels == i), 1] = Hex_to_RGB(hex_color_dict[i])[1]
        X_result[np.where(labels == i), 2] = Hex_to_RGB(hex_color_dict[i])[2]

    X_result = np.reshape(X_result, (row, col, 3))

    return X_result

def Cal_INDEX(x):

    x=x.astype('float')

    x[x>10000]=10000
    x[x<0]=0

    B = x[1, :, :]
    R = x[3, :, :]
    G = x[2, :, :]
    Nir = x[7, :, :]  # TM4
    Mir = x[10, :, :]  # TM5
    SWir= x[11,:,:]

    NDWI = (G - Nir) / (G + Nir)
    NDVI = (Nir - R) / (Nir + R)
    #NDSI = (Mir-Nir) / (Mir+Nir)
    NBI = R * SWir / Nir
    MSI = SWir / Nir
    BSI = ((Mir + R) - (Nir + B)) / ((Mir + R) + (Nir + B))

    return NDWI,NDVI,NBI, MSI,BSI

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Predicting")

    parser.add_argument('--model', type=str, default='deeplabv3',choices=['deeplabv3','unet'],
                        help='model name (default: deeplabv3)')
    parser.add_argument('--backbone', type=str, default='resnet',choices=['resnet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--depth', type=int, default=None, help='to choos which model depth(default: 50)')
    parser.add_argument('--out-stride', type=int, default=16, help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default=None,choices=['val', 'tes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--nclass', type=int, default=10, help='number of classes')
    parser.add_argument('--bsz', type=int, default=256, help='base image size')
    parser.add_argument('--csz', type=int, default=224, help='crop image size')
    parser.add_argument('--rsz', type=int, default=256, help='resample image size')
    parser.add_argument('--oly-s1', action='store_true', default=False, help='only use s1 data')
    parser.add_argument('--oly-s2', action='store_true', default=False, help='only use s2 data')
    parser.add_argument('--scale', type=str, default='std', choices=['std', 'norm'],
                        help='how to scale in preprocessing')

    parser.add_argument('--rgb', action='store_true', default=False, help='data augmentation')
    parser.add_argument('--denoise', action='store_true', default=False, help='data augmentation')
    parser.add_argument('--dehaze', action='store_true', default=False, help='data augmentation')
    parser.add_argument('--rule', type=str, default=None, choices=['dw_jiu', 'dw_new'], help='label filter')

    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=True,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--batch-size', type=int, default=2,metavar='N',
                        help='input batch size for training (default: auto)')
    parser.add_argument('--pre-batch-size', type=int, default=4,
                        metavar='N', help='input batch size for testing (default: auto)')
    parser.add_argument('--crf', action='store_true', default=False,help='crf postprocessing')
    parser.add_argument('--mode', type=str, default='none', choices=['soft','hard','none'], help='voting method')

    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    # checking point
    parser.add_argument('--resume', type=str, default='/data/PreTrainedModel/DSM/3446.pth.tar',
                        help='put the path to resuming file if needed')
    # output dir
    parser.add_argument('--dir', type=str, default=None,
                        help='folder of prediction of test dataset')
    parser.add_argument('--export', type=str, default=None, choices=['image', 'prob'],
                        help='export data category')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.batch_size is None:
        args.batch_size = 1

    if args.pre_batch_size is None:
        args.pre_batch_size = args.batch_size

    if args.oly_s1 and not args.oly_s2:
        print('Only use s1 SAR data!')
    elif not args.oly_s1 and args.oly_s2 and not args.rgb:
        print('Only use s2 MSI data!')
    elif not args.oly_s1 and args.oly_s2 and args.rgb:
        print('Only use s2 RGB data!')
    elif not args.oly_s1 and not args.oly_s2 and not args.rgb:
        print('Using s1 and s2 data in the same time!')
    elif not args.oly_s1 and not args.oly_s2 and args.rgb:
        print('Using s1 and s2 rgb data in the same time!')
    else:
        raise NotImplementedError

    outputdir=args.dir+'output/'
    visdir=args.dir+'vis/'
    probdir = args.dir + 'prob/'

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    if not os.path.exists(visdir):
        os.makedirs(visdir)

    if not os.path.exists(probdir):
        os.makedirs(probdir)

    ########################################## model ###########################################

    # Define network
    if args.model == 'deeplabv3':
        model = DeepLab(args,num_classes=args.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn,
                        depth=args.depth)

    if args.model == 'unet':
        model = Unet(args, num_classes=args.nclass, depth=args.depth)

    # Using cuda
    model = torch.nn.DataParallel(model)
    patch_replication_callback(model)
    model = model.cuda()

    if args.resume is not None:
        if not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)

        if args.cuda:
            model.module.load_state_dict(checkpoint['teacher_state_dict'])
        else:
            model.load_state_dict(checkpoint['teacher_state_dict'])

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))

    model.eval()

    ###################################### prepare data ########################################

    if args.dataset=='val':
        base_dir = '/data/PublicData/DF2020/val/'
        s1_pre, s2_pre, lc_pre=load_valdata(base_dir)
        print('{} val images for prediction'.format(s1_pre.shape[0]))


    if args.dataset=='tes':

        base_dir = '/data/PublicData/DF2020/test_track1/'
        s1_pre, s2_pre, lc_pre = load_tesdata(base_dir)

        print('{} tes images for prediction'.format(s1_pre.shape[0]))

    pre_dataset = DF2020(args, s1_pre, s2_pre, lc_pre, split='pre')

    pre_loader = DataLoader(dataset=pre_dataset, batch_size=args.pre_batch_size,shuffle=False)

    tbar = tqdm(pre_loader, desc='\r')

    evaluator = Evaluator(args.nclass+1)

    pre_prob = np.array([s1_pre.shape[0], 10, 256, 256])

    for i, (x1,x2,y,index) in enumerate(tbar):

        x1_ori=np.array(x1).transpose(0, 2, 3, 1) # N,H,W,C
        x2_ori=np.array(x2).transpose(0, 2, 3, 1)

        # x1
        data1_1 = x1_ori.copy()
        data1_2 = np.rot90(data1_1, 1, (1, 2)).copy()
        data1_3 = np.rot90(data1_1, 2, (1, 2)).copy()
        data1_4 = np.rot90(data1_1, 3, (1, 2)).copy()

        x1_hinv = x1_ori[:, :, ::-1, :].copy()

        data1_5 = x1_hinv.copy()
        data1_6 = np.rot90(x1_hinv, 1, (1, 2)).copy()
        data1_7 = np.rot90(x1_hinv, 2, (1, 2)).copy()
        data1_8 = np.rot90(x1_hinv, 3, (1, 2)).copy()

        data1_9 = np.concatenate([data1_1, data1_2, data1_3, data1_4,
                                  data1_5, data1_6, data1_7, data1_8], axis=0)
        # x2
        data2_1 = x2_ori.copy()
        data2_2 = np.rot90(data2_1, 1, (1, 2)).copy()
        data2_3 = np.rot90(data2_1, 2, (1, 2)).copy()
        data2_4 = np.rot90(data2_1, 3, (1, 2)).copy()

        x2_hinv = x2_ori[:, :, ::-1, :].copy()

        data2_5 = x2_hinv.copy()
        data2_6 = np.rot90(x2_hinv, 1, (1, 2)).copy()
        data2_7 = np.rot90(x2_hinv, 2, (1, 2)).copy()
        data2_8 = np.rot90(x2_hinv, 3, (1, 2)).copy()

        data2_9 = np.concatenate([data2_1, data2_2, data2_3, data2_4,
                                  data2_5, data2_6, data2_7, data2_8], axis=0)

        if args.cuda:
            data1_9 = data1_9.transpose(0, 3, 1, 2)  # (8,C,H,W)
            data2_9 = data2_9.transpose(0, 3, 1, 2)

            data1_9 = torch.from_numpy(data1_9).cuda()
            data2_9 = torch.from_numpy(data2_9).cuda()

        with torch.no_grad():

            output = model(data1_9,data2_9)

        output = F.softmax(output, dim=1)

        pred = output.data.cpu().numpy()#(8,10,H,W)

        pred[:,[2,7],:,:]=0

        if args.mode == 'hard':

            if not args.oly_s1 and args.crf:
                label_temp=np.zeros([8,pred.shape[-2],pred.shape[-1]])
                for j in range(pred.shape[0]):
                    _,label_temp[j]=CRF(pred[j],data2_9[j,[3,2,1],:,:].cpu().numpy())
            else:
                label_temp = np.argmax(pred, axis=1)  # (8,H,W)

            #############反变换################

            label1 = label_temp[0].copy()  # (1,H,W)
            label2 = np.rot90(label_temp[1], -1, (0, 1)).copy()
            label3 = np.rot90(label_temp[2], -2, (0, 1)).copy()
            label4 = np.rot90(label_temp[3], -3, (0, 1)).copy()

            label5 = label_temp[4][:, ::-1].copy()
            label6 = np.rot90(label_temp[5], -1, (0, 1))[:, ::-1].copy()
            label7 = np.rot90(label_temp[6], -2, (0, 1))[:, ::-1].copy()
            label8 = np.rot90(label_temp[7], -3, (0, 1))[:, ::-1].copy()

            ##############投票################

            label_mat_3 = np.concatenate(
                [np.expand_dims(label1, 0), np.expand_dims(label2, 0), np.expand_dims(label3, 0),
                 np.expand_dims(label4, 0), np.expand_dims(label5, 0), np.expand_dims(label6, 0),
                 np.expand_dims(label7, 0), np.expand_dims(label8, 0)], axis=0)  # (8,H,W)

            label = np.zeros((256 * 256,))
            label_mat_3 = np.reshape(label_mat_3, [8, 256 * 256])
            for m in range(256 * 256):
                temp = label_mat_3[:, m]
                label[m] = np.argmax(np.bincount(temp))

            label = np.reshape(label, [256, 256])

        elif args.mode == 'soft':
            label_temp = pred

            #############反变换################

            label1 = label_temp[0].copy()  # (10,H,W)
            label2 = np.rot90(label_temp[1], -1, (1, 2)).copy()
            label3 = np.rot90(label_temp[2], -2, (1, 2)).copy()
            label4 = np.rot90(label_temp[3], -3, (1, 2)).copy()

            label5 = label_temp[4][:, :, ::-1].copy()
            label6 = np.rot90(label_temp[5], -1, (1, 2))[:, :, ::-1].copy()
            label7 = np.rot90(label_temp[6], -2, (1, 2))[:, :, ::-1].copy()
            label8 = np.rot90(label_temp[7], -3, (1, 2))[:, :, ::-1].copy()

            ##############投票################

            label_mat = label1 + label2 + label3 + label4 + label5 + label6 + label7 + label8  # (10,H,W)

            if not args.oly_s1 and args.crf:
                if args.rgb:
                    _, label = CRF(label_mat / 8, x2.squeeze().numpy())
                else:
                    _, label = CRF(label_mat / 8, x2[:, [3, 2, 1], :, :].squeeze().numpy())
            else:
                label = np.argmax(label_mat, 0)

        elif args.mode == 'none':

            label_mat=pred[0,:,:,:]  # (10,H,W)

            if not args.oly_s1 and args.crf:
                if args.rgb:
                    _, label = CRF(label_mat / 8, x2.squeeze().numpy())
                else:
                    _, label = CRF(label_mat, x2[:,[3, 2, 1],:,:].squeeze().numpy())
            else:
                label = np.argmax(label_mat, 0)

        else:
            raise  NotImplementedError

        if args.export == 'image':

            im = np.uint8(label + 1)

            filename=os.path.basename(lc_pre[index])
            former = filename.split('lc')[0]
            latter = filename.split('lc')[1]

            f=base_dir+'s2_0/'+former+'s2'+latter

            with rasterio.open(f) as patch:
                x2_img = patch.read(list(range(1, 14)))

            #print(x2_img.shape)

            NDWI,NDVI,NBI, MSI,BSI=Cal_INDEX(x2_img)

            y_tmp = y[0,0,:,:].cpu().numpy().copy()

            # im[np.where(NDWI>0)]=10
            # label[np.where(NDWI>0)]=9

            im_tmp = im.copy()
            y_pre_tmp = label.copy()

            # grassland

            im[np.where((NDWI < 0) & (NDVI > 0.4) & (NDVI < 0.6) &
                        ((y_tmp == 2) | (y_tmp == 3) | (im_tmp == 5)) &
                        (np.sum(NDWI > 0) < 2000))] = 4

            label[np.where((NDWI < 0) & (NDVI > 0.4) & (NDVI < 0.6) &
                           ((y_tmp == 2) | (y_tmp == 3) | (y_pre_tmp == 4)) &
                           (np.sum(NDWI > 0) < 2000))] = 3

            # wetland

            im[np.where((NDVI > 0.6) & (NDVI < 0.75) &
                        ((y_tmp == 4) | (im_tmp == 4)) &
                        (np.sum(NDWI > 0) > 4000))] = 5

            label[np.where((NDVI > 0.6) & (NDVI < 0.75) &
                           ((y_tmp == 4) | (y_pre_tmp == 3)) &
                           (np.sum(NDWI > 0) > 4000))] = 4

            # forest

            im[np.where((NDWI < 0) & (NDVI > 0.75) &
                        ((y_tmp == 2) | (y_tmp == 0) | (im_tmp == 4) | (im_tmp == 5)))] = 1
            label[np.where((NDWI < 0) & (NDVI > 0.75) &
                           ((y_tmp == 2) | (y_tmp == 0) | (y_pre_tmp == 3) | (y_pre_tmp == 4)))] = 0

            im[np.where((im_tmp == 5) & (np.sum(NDWI > 0) < 1000))] = 1
            label[np.where((y_pre_tmp == 4) & (np.sum(NDWI > 0) < 1000))] = 0

            # cropland

            im[np.where((NDWI < 0) & (NDVI < 0.4) & (NDVI > 0.2) & (MSI > 1) & (MSI < 1.5) &
                        ((y_tmp == 5) | (y_tmp == 2)))] = 6
            label[np.where((NDWI < 0) & (NDVI < 0.4) & (NDVI > 0.2) & (MSI > 1) & (MSI < 1.5) &
                           ((y_tmp == 5) | (y_tmp == 2)))] = 5

            # urban

            im[np.where((NDWI < 0) & (NDVI < 0.2) & (NDVI > 0) & (BSI > -0.4) & (y_tmp == 6))] = 7
            label[np.where((NDWI < 0) & (NDVI < 0.2) & (NDVI > 0) & (BSI > -0.4) & (y_tmp == 6))] = 6

            # barren

            im[np.where((NDWI < 0) & (NBI > 750) & (NDVI < 0.4) & (NDVI > 0) &
                        (y_tmp != 5) & (y_tmp != 6) & (im_tmp != 6) & (im_tmp != 7) &
                        ((im_tmp == 4) | (im_tmp == 2)))] = 9
            label[np.where((NDWI < 0) & (NBI > 750) & (NDVI < 0.4) & (NDVI > 0) &
                           (y_tmp != 5) & (y_tmp != 6) & (y_pre_tmp != 5) & (y_pre_tmp != 6) &
                           ((y_pre_tmp == 3) | (y_pre_tmp == 1)))] = 8

            # shrubland

            im[np.where((NDWI < 0) & (y_tmp == 1) & (im_tmp == 2))] = 2
            label[np.where((NDWI < 0) & (y_tmp == 1) & (y_pre_tmp == 1))] = 1

            imsave(outputdir + former + 'dfc' + latter, im)
            im_rgb = Image.fromarray(np.uint8(DrawResult(label.reshape(-1), x2.shape[-2], x2.shape[-1])))
            im_rgb.save(visdir + former + 'dfc' + latter[:-4] + '_vis.png')

        elif args.export == 'prob':

            pre_prob[[index]] = pred[[0]]

        else:
            raise NotImplementedError

        target = y[:, 0, :, :].cpu().numpy()  # batch_size * 256 * 256
        # # Add batch sample into evaluator
        evaluator.add_batch(target, label[np.newaxis, :, :])

    AA = evaluator.pre_Pixel_Accuracy_Class()

    print('AVERAGE ACCURACY of {} DATASET: {}'.format(str(args.dataset), AA))

    print('ACCURACY IN EACH CLASSES:', np.diag(evaluator.confusion_matrix) / evaluator.confusion_matrix.sum(axis=1))

    if args.export == 'image':

        print('IMAGE EXPORTED finished!')

    elif args.export == 'prob':

        np.save(probdir + 'DFC2020_tes_' + str(args.model) + '_' + str(args.backbone) + '_.npy', pre_prob)

        print('PROBABILITY EXPORTED finished!')

    print('Prediction finished!')

if __name__ == "__main__":
    main()
    # model=DeepLab(num_classes=3)
    # model.eval()
    # input=torch.randn(1,3,256,256)
    # output=model(input)
    # print(output.shape)
