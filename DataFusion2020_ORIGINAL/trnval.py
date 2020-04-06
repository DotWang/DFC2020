import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from utils.lr_scheduler import LR_Scheduler
from torch.autograd import Variable
from module import *

class operater(object):
    def __init__(self, args, model,trn_loader,val_loader,chk_loader,optimizer):

        self.args = args
        self.model=model
        self.train_loader = trn_loader
        self.val_loader = val_loader
        self.chk_loader = chk_loader
        self.optimizer = optimizer
        # Define Evaluator
        self.evaluator = Evaluator(args.nclass)
        # Define lr scheduler
        # self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
        #                          args.epochs, len(trn_loader))
        self.scheduler=torch.optim.lr_scheduler.MultiStepLR(self.optimizer,milestones=[3,6,9], gamma=0.5)
        self.wait_epoches=10
        self.best_pred = 0
        self.init_weight=0.98
        # Define Saver
        self.saver = Saver(self.args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        self.evaluator = Evaluator(self.args.nclass)

    def training(self,epoch,args):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        #w1 = 0.2 + 0.5 * (self.init_weight - 0.2) * (1 + np.cos(epoch * np.pi / args.epochs))
        print('Learning rate:', self.optimizer.param_groups[0]['lr'])
        for i, (x1,x2,y,index) in enumerate(tbar):
            x1=Variable(x1)
            x2=Variable(x2)
            #y_cls=Seg2cls(args,y)#图像级标签，N,1,1,C
            if self.args.cuda:
                x1, x2 = x1.cuda(),x2.cuda()
            #self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(x1,x2)

            output = F.softmax(output, dim=1)

            # loss_ce=CELossLayer(self.args,output,y)
            # #print('ce loss', loss_ce)
            #
            # loss_focal = FocalLossLayer(self.args,output, y)
            #print('focal loss', loss_focal)

            loss_lovasz = LovaszLossLayer(output,y)
            #print('lovasz loss', loss_lovasz)

            # self.writer.add_scalar('train/ce_loss_iter', loss_focal.item(), i + num_img_tr * epoch)
            # self.writer.add_scalar('train/focal_loss_iter', loss_focal.item(), i + num_img_tr * epoch)
            self.writer.add_scalar('train/lovasz_loss_iter', loss_lovasz.item(), i + num_img_tr * epoch)

            #loss = w1 * loss_ce + (0.5 - 0.5 * w1) * loss_focal + (0.5 - 0.5 * w1) * loss_lovasz

            loss = loss_lovasz

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            #Show 10 * 3 inference results each epoch
            if i % 10 == 0:
                global_step = i + num_img_tr * epoch
                if self.args.oly_s1 and not self.args.oly_s2:
                    self.summary.visualize_image(self.writer, self.args.dataset, x1[:,[0],:,:], y, output, global_step)
                elif not self.args.oly_s1:
                    if self.args.rgb:
                        self.summary.visualize_image(self.writer, self.args.dataset, x2, y, output, global_step)
                    else:
                        self.summary.visualize_image(self.writer, self.args.dataset, x2[:,[2,1,0],:,:], y,output,global_step)
                else:
                    raise NotImplementedError
        self.scheduler.step()
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + y.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, is_best)

    def validation(self,epoch, args):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        for i, (x1,x2,y,index) in enumerate(tbar):

            if self.args.cuda:
                x1, x2 = x1.cuda(),x2.cuda()
            with torch.no_grad():
                output = self.model(x1, x2)

            pred = output.data.cpu().numpy()
            pred[:,[2,7],:,:]=0
            target = y[:,0,:,:].cpu().numpy()  # batch_size * 256 * 256
            pred = np.argmax(pred, axis=1)  # batch_size * 256 * 256
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        OA = self.evaluator.Pixel_Accuracy()
        AA = self.evaluator.val_Pixel_Accuracy_Class()
        self.writer.add_scalar('val/OA', OA, epoch)
        self.writer.add_scalar('val/AA', AA, epoch)

        print('AVERAGE ACCURACY:', AA)

        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + y.data.shape[0]))

        new_pred = AA
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

