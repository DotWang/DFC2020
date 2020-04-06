import numpy as np
import torch
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, create_pairwise_bilateral,create_pairwise_gaussian, unary_from_softmax


pre_hex_color_dict={10:'000000',0:'009900',1:'c6b044',2:'fbff13',3:'b6ff05',4:'27ff87',5:'c24f44',
                    6:'a5a5a5',7:'69fff8',8:'f9ffa4',9:'1c0dff'}

trn_hex_color_dict={10:'000000',0:'009900',1:'c6b044',2:'b6ff05',3:'27ff87',4:'c24f44',
                    5:'a5a5a5',6:'f9ffa4',7:'1c0dff'}

def Hex_to_RGB(str):
    r = int(str[0:2],16)
    g = int(str[2:4],16)
    b = int(str[4:6],16)
    return [r,g,b]

def decode_seg_map_sequence(label_masks, dataset='S1S2'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask,8,trn_hex_color_dict, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def decode_segmap(label_mask, n_classes, hex_color_dict, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = Hex_to_RGB(hex_color_dict[ll])[0]
        g[label_mask == ll] = Hex_to_RGB(hex_color_dict[ll])[1]
        b[label_mask == ll] = Hex_to_RGB(hex_color_dict[ll])[2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0

    return rgb


def CRF(x,image):
    unary = unary_from_softmax(x)#C,H,W
    unary = np.ascontiguousarray(unary)
    d = dcrf.DenseCRF(image.shape[-2] * image.shape[-1], 10)
    d.setUnaryEnergy(unary)

    gau_feats = create_pairwise_gaussian(sdims=(3, 3), shape=image.shape[-2:])
    d.addPairwiseEnergy(gau_feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    bil_feats = create_pairwise_bilateral(sdims=(10, 10), schan=[5],
                                          img=np.array(image).transpose(1, 2, 0), chdim=2)
    d.addPairwiseEnergy(bil_feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(5)
    x = np.argmax(Q, axis=0).reshape(image.shape[-2], image.shape[-1])
    Q=np.array(Q)
    Q=Q.reshape(-1,image.shape[-2], image.shape[-1])
    return Q,x
