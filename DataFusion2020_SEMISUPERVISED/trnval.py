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
    def __init__(self, args, student_model,teacher_model,src_loader,trg_loader,val_loader,optimizer,teacher_optimizer):

        self.args = args
        self.student_model=student_model
        self.teacher_model=teacher_model
        self.src_loader = src_loader
        self.trg_loader = trg_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.teacher_optimizer=teacher_optimizer
        # Define Evaluator
        self.evaluator = Evaluator(args.nclass)
        # Define lr scheduler
        # self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
        #                          args.epochs, len(trn_loader))
        #self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[3, 6, 9, 12], gamma=0.5)
        #ft
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20], gamma=0.5)
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
        self.student_model.train()
        self.teacher_model.train()
        num_src = len(self.src_loader)
        num_trg = len(self.trg_loader)
        num_itr = np.maximum(num_src,num_trg)
        tbar = tqdm(range(1,num_itr+1))
        #w1 = 0.2 + 0.5 * (self.init_weight - 0.2) * (1 + np.cos(epoch * np.pi / args.epochs))
        print('Learning rate:', self.optimizer.param_groups[0]['lr'])
        iter_src=iter(self.src_loader)
        iter_trg=iter(self.trg_loader)
        for i in tbar:

            src_x1,src_x2,src_y,src_idx=iter_src.next()
            trg_x1,trg_x2,trg_y,trg_idx=iter_trg.next()

            if i % num_src==0:
                iter_src=iter(self.src_loader)

            if self.args.cuda:
                src_x1, src_x2 = src_x1.cuda(),src_x2.cuda()
                trg_x1, trg_x2 = trg_x1.cuda(),trg_x2.cuda()

            self.optimizer.zero_grad()

            # train with source

            _,_,src_output = self.student_model(src_x1,src_x2)

            src_output = F.softmax(src_output, dim=1)


            # CE loss of supervised data

            #loss_ce=CELossLayer(src_output,src_y)

            # #print('ce loss', loss_ce)
            # Focal loss of supervised data
            loss_focal = FocalLossLayer(src_output,src_y)
            #print('focal loss', loss_focal)

            loss_val_lovasz = LovaszLossLayer(src_output,src_y)
            #print('lovasz loss', loss_lovasz)

            if epoch > 3:
                loss_su=loss_val_lovasz + loss_focal
            else:
                loss_su=loss_val_lovasz + loss_focal

            # train with target

            trg_x1_s = trg_x1 + torch.randn(trg_x1.size()).cuda() * self.args.noise
            trg_x1_t = trg_x1 + torch.randn(trg_x1.size()).cuda() * self.args.noise

            trg_x2_s = trg_x2 + torch.randn(trg_x2.size()).cuda() * self.args.noise
            trg_x2_t = trg_x2 + torch.randn(trg_x2.size()).cuda() * self.args.noise

            _,_,trg_predict_s = self.student_model(trg_x1_s,trg_x2_s)

            _,spatial_mask_prob,trg_predict_t = self.teacher_model(trg_x1_t,trg_x2_t)

            trg_predict_s = F.softmax(trg_predict_s, dim=1)
            trg_predict_t = F.softmax(trg_predict_t, dim=1)

            loss_tes_lovasz = LovaszLossLayer(trg_predict_s,trg_y)

            # spatial mask

            #channel_mask = channel_mask_prob > args.attention_threshold
            spatial_mask = spatial_mask_prob > args.attention_threshold

            spatial_mask = spatial_mask.float()

            #spatial_mask = spatial_mask.permute(0,2,3,1)# N,H,W,C

            #channel_mask = channel_mask.float()
            #spatial_mask = spatial_mask.view(-1)

            num_pixel = spatial_mask.shape[0]*spatial_mask.shape[-2]*spatial_mask.shape[-1]

            mask_num_rate = torch.sum(spatial_mask).float() / num_pixel

            # trg_output_s = trg_output_s.permute(0, 2, 3, 1)#N,H,W,C
            # trg_output_t = trg_output_t.permute(0, 2, 3, 1)

            #trg_output_s = trg_output_s * channel_mask
            trg_predict_s = trg_predict_s * spatial_mask

            #trg_output_t = trg_output_t * channel_mask
            trg_predict_t = trg_predict_t * spatial_mask

            # trg_output_s = trg_output_s.contiguous().view(-1, self.args.nclass)
            # trg_output_s = trg_output_s[spatial_mask]
            #
            # trg_output_t = trg_output_t.contiguous().view(-1, self.args.nclass)
            # trg_output_t = trg_output_t[spatial_mask]

            # consistency loss

            loss_con = ConsistencyLossLayer(trg_predict_s,trg_predict_t)

            if mask_num_rate==0.:

                loss_con = torch.tensor(0.).float().cuda()

            loss=loss_su + self.args.con_weight*loss_con + self.args.teslab_weight*loss_tes_lovasz

            #self.writer.add_scalar('train/ce_loss_iter', loss_ce.item(), i + num_itr * epoch)
            self.writer.add_scalar('train/focal_loss_iter', loss_focal.item(), i + num_itr * epoch)
            self.writer.add_scalar('train/supervised_loss_iter', loss_su.item(), i + num_itr * epoch)
            self.writer.add_scalar('train/consistency_loss_iter', loss_con.item(), i + num_itr * epoch)
            self.writer.add_scalar('train/teslab_loss_iter', loss_tes_lovasz.item(), i + num_itr * epoch)
            #loss = w1 * loss_ce + (0.5 - 0.5 * w1) * loss_focal + (0.5 - 0.5 * w1) * loss_lovasz

            loss.backward()
            self.optimizer.step()
            self.teacher_optimizer.step()

            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_itr * epoch)

            #Show 10 * 3 inference results each epoch
            if i % 10 == 0:
                global_step = i + num_itr * epoch
                if self.args.oly_s1 and not self.args.oly_s2:
                    self.summary.visualize_image(self.writer, self.args.dataset, src_x1[:,[0],:,:], trg_x1[:,[0],:,:],
                                                 src_y, src_output, trg_predict_s, trg_predict_t, trg_y, global_step)
                elif not self.args.oly_s1:
                    if self.args.rgb:
                        self.summary.visualize_image(self.writer, self.args.dataset, src_x2, trg_x2,
                                                 src_y, src_output, trg_predict_s, trg_predict_t, trg_y,global_step)
                    else:
                        self.summary.visualize_image(self.writer, self.args.dataset, src_x2[:,[2,1,0],:,:], trg_x2[:,[2,1,0],:,:],
                                                 src_y, src_output, trg_predict_s, trg_predict_t, trg_y, global_step)
                else:
                    raise NotImplementedError

        self.scheduler.step()
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + src_y.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'student_state_dict': self.student_model.module.state_dict(),
                'teacher_state_dict': self.teacher_model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, is_best)

    def validation(self,epoch, args):
        self.teacher_model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        for i, (x1,x2,y,index) in enumerate(tbar):

            if self.args.cuda:
                x1, x2 = x1.cuda(),x2.cuda()
            with torch.no_grad():
               _,_, output = self.teacher_model(x1, x2)

            output = F.softmax(output,dim=1)
            pred = output.data.cpu().numpy()
            #pred[:,[2,7],:,:]=0
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
                'student_state_dict': self.student_model.module.state_dict(),
                'teacher_state_dict': self.teacher_model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


