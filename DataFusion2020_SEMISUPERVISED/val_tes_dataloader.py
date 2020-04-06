import  os
from glob import glob
import numpy as np
import rasterio
from enum import Enum
from tqdm import tqdm


def load_tesdata(base_dir):

    s1_dir=glob(base_dir+'s1_0'+'/'+'*.tif')
    s2_dir=[]
    lc_dir=[]
    for i in tqdm(range(len(s1_dir))):
        s1_filename=os.path.basename(s1_dir[i])
        former=s1_filename.split('_s1')[0]
        ID=s1_filename.split('_s1')[1]
        s2_dir.append(base_dir+'s2_0/'+former+'_s2'+ID)
        lc_dir.append(base_dir + 'lc_fake/' + former + '_lc' + ID)

    s1_dir=np.array(s1_dir)
    s2_dir=np.array(s2_dir)
    lc_dir=np.array(lc_dir)

    return s1_dir,s2_dir,lc_dir

def load_valdata(base_dir):

    s1_dir=glob(base_dir+'s1_validation'+'/'+'*.tif')
    s2_dir=[]
    lc_dir=[]
    for i in tqdm(range(len(s1_dir))):
        s1_filename=os.path.basename(s1_dir[i])
        former=s1_filename.split('_s1')[0]
        ID=s1_filename.split('_s1')[1]
        s2_dir.append(base_dir+'s2_validation/'+former+'_s2'+ID)
        lc_dir.append(base_dir + 'dfc_validation/' + former + '_dfc' + ID)

    s1_dir=np.array(s1_dir)
    s2_dir=np.array(s2_dir)
    lc_dir=np.array(lc_dir)

    return s1_dir,s2_dir,lc_dir


# s1_data=np.zeros([len(s1_dir),2,256,256])
# s2_data=np.zeros([len(s1_dir),13,256,256])
# lc_data=np.zeros([len(s1_dir),4,256,256])
#
# for i in tqdm(range(len(s1_dir))):
#
#     with rasterio.open(s1_dir[i]) as patch:
#         s1_tmp = patch.read(list(range(1,3)))
#
#     if len(s1_tmp.shape) == 2:
#         s1_tmp = np.expand_dims(s1_tmp, axis=0)
#
#     s1_data[i]=s1_tmp
#
#     with rasterio.open(s2_dir[i]) as patch:
#         s2_tmp = patch.read(list(range(1,14)))
#
#     if len(s2_tmp.shape) == 2:
#         s2_tmp = np.expand_dims(s2_tmp, axis=0)
#
#     s2_data[i] = s2_tmp
#
#     with rasterio.open(lc_dir[i]) as patch:
#         lc_tmp = patch.read(list(range(1,5)))
#
#     if len(lc_tmp.shape) == 2:
#         lc_tmp = np.expand_dims(lc_tmp, axis=0)
#
#     lc_data[i] = lc_tmp
#
# print('s1:',s1_data.shape)
# print('s2:',s2_data.shape)
# print('lc:',lc_data.shape)

#np.save('/data/PublicData/DF2020/val/s1_val.npy', s1_data)
#np.save('/data/PublicData/DF2020/val/s2_val.npy', s2_data)
#np.save('/data/PublicData/DF2020/val/lc_val.npy', lc_data)
#lc_dir=np.array(lc_dir)
#np.save('/data/PublicData/DF2020/val/lc_dir.npy', lc_dir)


