import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.networks import *
import rasterio
from DFC2020 import DF2020
from preparedata import load_data
from trnval import operater
from val_tes_dataloader import *

parser = argparse.ArgumentParser(description="PyTorch DeeplabFPN Training")
parser.add_argument('--model', type=str, default='deeplabv3',choices=['deeplabv3','deeplabfpn','vgg16_sec','unet'],
                        help='model name (default: deeplabv3)')
parser.add_argument('--backbone', type=str, default='resnet',
                    choices=['resnet','xception', 'drn', 'mobilenet', 'vgg', 'densenet', 'inception', 'squeezenet'],
                    help='backbone name (default: resnet)')
parser.add_argument('--out-stride', type=int, default=16,help='network output stride (default: 8)')
parser.add_argument('--depth', type=int, default=None,help='to choos which model depth(default: 50)')
# resnet:50, 101   vgg:16, 17, 19, 20   densenet: 121, 169, 201, 161   squeezenet:1, 2
# dataset need to change
parser.add_argument('--dataset', type=str, default='S1S2',choices=['S1S2'],help='dataset name (default: pascal)')
parser.add_argument('--nclass',type=int,default=10,help='number of classes')
parser.add_argument('--trn_ratio',type=float,default=0.1,help='training samples proportion')
parser.add_argument('--val_ratio',type=float,default=0.02,help='valing samples proportion')
parser.add_argument('--bsz', type=int, default=256,help='base image size')
parser.add_argument('--csz', type=int, default=224,help='crop image size')
parser.add_argument('--rsz', type=int, default=256,help='resample image size')
parser.add_argument('--oly-s1', action='store_true', default=False,help='only use s1 data')
parser.add_argument('--oly-s2', action='store_true', default=False,help='only use s2 data')
parser.add_argument('--scale', type=str, default='std', choices=['std','norm'],help='how to scale in preprocessing')
parser.add_argument('--aug', action='store_true', default=False,help='data augmentation')
parser.add_argument('--rgb', action='store_true', default=False,help='data augmentation')
parser.add_argument('--denoise', action='store_true', default=False,help='data augmentation')
parser.add_argument('--dehaze', action='store_true', default=False,help='data augmentation')
parser.add_argument('--rule',type=str,default=None,choices=['dw_jiu','dw_new'],help='label filter')
parser.add_argument('--extra_data',type=str,default=None,choices=['val','tes','oly_val','oly_tes','tes_val'],help='label filter')

# training hyper params
parser.add_argument('--epochs', type=int, default=None, metavar='N',help='number of epochs to train (default: auto)')
parser.add_argument('--start_epoch', type=int, default=0,metavar='N', help='start epochs (default:0)')
parser.add_argument('--batch-size', type=int, default=128,metavar='N', help='input batch size for training (default: auto)')
parser.add_argument('--val-batch-size', type=int, default=2,metavar='N', help='input batch size for testing (default: auto)')
parser.add_argument('--sync-bn', type=bool, default=False,help='whether to use sync bn (default: auto)')
parser.add_argument('--freeze-bn', type=bool, default=False,help='whether to freeze bn parameters (default: False)')

# optimizer params
parser.add_argument('--lr', type=float, default=None, metavar='LR',help='learning rate (default: auto)')
parser.add_argument('--lr-scheduler', type=str, default='multistep',choices=['poly', 'step', 'cos','none','multistep'],
                    help='lr scheduler mode: (default: poly)')
parser.add_argument('--momentum', type=float, default=0.9,metavar='M', help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=5e-4,metavar='M', help='w-decay (default: 5e-4)')
parser.add_argument('--nesterov', action='store_true', default=False,help='whether use nesterov (default: False)')
parser.add_argument('--weighted',action='store_true', default=False,help='weighted')
parser.add_argument('--ft', action='store_true', default=False,help='finetuning on a different dataset')

# cuda, seed and logging
parser.add_argument('--no-cuda', action='store_true', default=False,help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
# checking point
parser.add_argument('--resume', type=str, default=None,help='put the path to resuming file if needed')
parser.add_argument('--checkname', type=str, default=None,help='set the checkpoint name')
# evaluation option
parser.add_argument('--eval-interval', type=int, default=1,help='evaluation interval (default: 1)')
parser.add_argument('--no-val', action='store_true', default=False,help='skip validation during training')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.sync_bn is None:
    args.sync_bn = False

# default settings for epochs, batch_size and lr
if args.epochs is None:
    epoches = {
        'S1S2': 200
    }
    args.epochs = epoches[args.dataset.lower()]

if args.batch_size is None:
    args.batch_size = 2

if args.val_batch_size is None:
    args.val_batch_size = args.batch_size

if args.checkname is None:
    args.checkname = str(args.model) + str(args.backbone)

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

########################################## model ###########################################

# Define network

if args.model == 'deeplabv3':
    model = DeepLab(args,num_classes=args.nclass,
                    backbone=args.backbone,
                    output_stride=args.out_stride,
                    sync_bn=args.sync_bn,
                    freeze_bn=args.freeze_bn,
                    depth=args.depth)
    Seed_HW = 64
if args.model == 'deeplabfpn':
    model = DeeplabFPN(args,num_classes=args.nclass,
                       backbone=args.backbone,
                       output_stride=args.out_stride,
                       sync_bn=args.sync_bn,
                       freeze_bn=args.freeze_bn,
                       depth=args.depth)
    Seed_HW = 128

if args.model == 'unet':
    model = Unet(args,num_classes=args.nclass,depth=args.depth)

    Seed_HW=100


if args.model == 'deeplabv3' or args.model == 'deeplabfpn' or args.model == 'unet':
    # train_params = [{'params': model.get_10x_lr_params(), 'lr': args.lr},
    # ]

    # Define Optimizer
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr=args.lr, momentum=args.momentum,
    #                             weight_decay=args.weight_decay, nesterov=args.nesterov)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=args.lr,
                                 weight_decay=args.weight_decay)

else:
    optimizer = torch.optim.SGD(model.parameters(),lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=args.nesterov)

# Using cuda

model = torch.nn.DataParallel(model)
patch_replication_callback(model)
model = model.cuda()

# Resuming checkpoint

if args.resume is not None:
    if not os.path.isfile(args.resume):
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    args.start_epoch = checkpoint['epoch']
    if args.cuda:
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    if not args.ft:
        optimizer.load_state_dict(checkpoint['optimizer'])
    # self.best_pred = checkpoint['best_pred']
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(args.resume, checkpoint['epoch']))

###################################### prepare data ########################################

# s1_trn,s2_trn,y_trn,s1_val,s2_val,y_val=load_data(args)

# if args.extra_data=='val':
#     base_dir = '/data/PublicData/DF2020/val/'
#     s1_pre, s2_pre, lc_pre = load_valtesdata(base_dir)
#     s1_trn = np.concatenate((s1_trn, s1_pre),axis=0)
#     s2_trn = np.concatenate((s2_trn, s2_pre), axis=0)
#     y_trn = np.concatenate((y_trn,lc_pre),axis=0)
#     print('Validation dataset added!')

if args.extra_data=='oly_val':
    base_dir = '/data/PublicData/DF2020/val/'
    s1_pre, s2_pre, lc_pre = load_valdata(base_dir)
    s1_trn = s1_pre.copy()
    s2_trn = s2_pre.copy()
    y_trn = lc_pre.copy()

    s1_val = s1_pre[:200]
    s2_val = s2_pre[:200]
    y_val = lc_pre[:200]

    print('Only use Validation dataset!')

elif args.extra_data=='oly_tes':
    base_dir = '/data/PublicData/DF2020/test_track1/'
    s1_pre, s2_pre, lc_pre = load_tesdata(base_dir)
    s1_trn = s1_pre.copy()
    s2_trn = s2_pre.copy()
    y_trn = lc_pre.copy()

    idx = np.arange(s1_trn.shape[0])
    np.random.shuffle(idx)

    s1_val = s1_trn[idx[:1000]]
    s2_val = s2_trn[idx[:1000]]
    y_val = y_trn[idx[:1000]]

    print('Only use Testing dataset!')

elif args.extra_data=='tes_val':

    base_dir = '/data/PublicData/DF2020/val/'
    s1_1, s2_1, lc_1 = load_valdata(base_dir)
    base_dir = '/data/PublicData/DF2020/test_track1/'
    s1_2, s2_2, lc_2 = load_tesdata(base_dir)

    s1_trn = np.concatenate((s1_1, s1_2),axis=0)
    s2_trn = np.concatenate((s2_1, s2_2), axis=0)
    y_trn = np.concatenate((lc_1,lc_2),axis=0)

    idx=np.arange(s1_trn.shape[0])
    np.random.shuffle(idx)

    s1_val = s1_trn[idx[:1000]]
    s2_val = s2_trn[idx[:1000]]
    y_val = y_trn[idx[:1000]]

    print('Using Testing and Validation dataset!')


print('{} training images'.format(s1_trn.shape[0]))
print('{} validation images'.format(s1_val.shape[0]))
print('Data preparing finished!')

trn_dataset=DF2020(args,s1_trn,s2_trn,y_trn,[],[],[],[],[],[],split='trn')
val_dataset=DF2020(args,s1_val,s2_val,y_val,[],[],[],[],[],[],split='val')

trn_dataloader=DataLoader(dataset=trn_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(range(s1_trn.shape[0])),drop_last=True)
val_dataloader=DataLoader(dataset=val_dataset,batch_size=args.val_batch_size)

########################## training & valing & saver ######################

torch.manual_seed(args.seed)
trainer = operater(args,model,trn_dataloader,val_dataloader,[],optimizer)
print('Starting Epoch:', args.start_epoch)
print('Total Epoches:', args.epochs)

i=0
for epoch in range(args.start_epoch, args.epochs):
    print()
    trainer.training(epoch,args)
    #chk_seeds[:,i,:,:]=trainer.saving(epoch,args)
    #i+=1
    if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
        trainer.validation(epoch,args)

trainer.writer.close()

print('Training finished!')

