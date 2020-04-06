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
    
    def visualize_image(self, writer, dataset, image, target, output, global_step):
        grid_image = make_grid(image[:3,:,:,:].clone().cpu().data*2.5, 3, normalize=True)
        writer.add_image('Image', grid_image, global_step)
        output[:, [2, 7], :, :] = 0
        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image('Predicted label', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image('Groundtruth label', grid_image, global_step)
    #     grid_image = make_grid(decode_seg_map_sequence(init_seed[:3].argmax(1).numpy(),
    #                                                    dataset=dataset), 3, normalize=False, range=(0, 255))
    #     writer.add_image('Init seed', grid_image, global_step)
    #     grid_image = make_grid(decode_seg_map_sequence(seedmap[:3].argmax(1).numpy(),
    #                                                    dataset=dataset), 3, normalize=False, range=(0, 255))
    #     writer.add_image('Grow seed', grid_image, global_step)
    #