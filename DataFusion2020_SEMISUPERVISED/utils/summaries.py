import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from utils.func import *

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer
    
    def visualize_image(self, writer, dataset, src_image, trg_image, src_target, src_output, trg_output_s, trg_output_t
                        , trg_target, global_step):

        grid_image = make_grid(src_image[:3,:,:,:].clone().cpu().data*2.5, 3, normalize=True)
        writer.add_image('Source Image', grid_image, global_step)

        grid_image = make_grid(trg_image[:3, :, :, :].clone().cpu().data * 2.5, 3, normalize=True)
        writer.add_image('Target Image', grid_image, global_step)

        #src_output[:, [2, 7], :, :] = 0
        grid_image = make_grid(decode_seg_map_sequence(torch.max(src_output[:3], 1)[1].detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image('Source Prediction in Student Network', grid_image, global_step)

        #trg_output_s[:, [2, 7], :, :] = 0
        grid_image = make_grid(decode_seg_map_sequence(torch.max(trg_output_s[:3], 1)[1].detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image('Target Prediction in Student Network', grid_image, global_step)

       # trg_output_t[:, [2, 7], :, :] = 0
        grid_image = make_grid(decode_seg_map_sequence(torch.max(trg_output_t[:3], 1)[1].detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image('Target Prediction in Teacher Network', grid_image, global_step)


        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(src_target[:3], 1).detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image('Source Groundtruth label', grid_image, global_step)

        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(trg_target[:3], 1).detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image('Target Groundtruth label', grid_image, global_step)
