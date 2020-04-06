import numpy as np
import torch
import cv2
from scipy.ndimage import gaussian_filter
import rasterio
import random
from PIL import Image, ImageOps, ImageFilter,ImageEnhance

def zmMinFilterGray(src, r=9):
    '''''æœ€å°å€¼æ»¤æ³¢ï¼Œræ˜¯æ»¤æ³¢å™¨åŠå¾„'''
    return cv2.erode(src, np.ones((2 * r - 1, 2 * r - 1)))


def guidedfilter(I, p, r, eps):
    '''''å¼•å¯¼æ»¤æ³¢ï¼Œç›´æŽ¥å‚è€ƒç½‘ä¸Šçš„matlabä»£ç '''
    height, width = I.shape
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b


def getV1(m, r, eps, w, maxV1):  # è¾“å…¥rgbå›¾åƒï¼Œå€¼èŒƒå›´[0,1]
    '''''è®¡ç®—å¤§æ°”é®ç½©å›¾åƒV1å’Œå…‰ç…§å€¼A, V1 = 1-t/A'''
    V1 = np.min(m, 2)  # å¾—åˆ°æš—é€šé“å›¾åƒ
    V1 = guidedfilter(V1, zmMinFilterGray(V1, 5), r, eps)  # ä½¿ç”¨å¼•å¯¼æ»¤æ³¢ä¼˜åŒ–
    bins = 2000
    ht = np.histogram(V1, bins)  # è®¡ç®—å¤§æ°”å…‰ç…§A
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()

    V1 = np.minimum(V1 * w, maxV1)  # å¯¹å€¼èŒƒå›´è¿›è¡Œé™åˆ¶

    return V1, A

lab_dict={1:1,2:1,3:1,4:1,5:1,6:2,7:2,8:31,9:32,10:4,11:5,12:6,14:6,13:7,15:8,16:9,17:10}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
class DF2020(torch.utils.data.Dataset):
    def __init__(self,args,s1,s2,y,s1_mean,s1_std,s2_mean,s2_std,s2_max,Seed_HW,split='trn'):
        self.args=args
        self.s1=s1
        self.s2=s2
        self.y=y
        self.s1_mean=s1_mean
        self.s1_std=s1_std
        self.s2_mean=s2_mean
        self.s2_std=s2_std
        self.s2_max=s2_max
        self.split=split
    def __len__(self):
        return len(self.s1)
    def __getitem__(self, index):
        fname1=self.s1[index]
        fname2=self.s2[index]
        x1,x2=self.read_data(fname1,fname2)
        x1=x1.transpose(1,2,0)#h,w,c
        x2=x2.transpose(1,2,0)
        x2_tmp=x2.copy()
        if self.args.rgb:
            x2=x2[:,:,[3,2,1]]
        else:
            x2 = x2[:, :, [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]]
        x1,x2=self.clean(x1,x2)
        if self.split=='trn':
            fname3 = self.y[index]
            with rasterio.open(fname3) as patch:
                y = patch.read(list(range(1, 5)))
            y=self.process_label(y)
            y=self.filter_label(x2_tmp,y)
            if self.args.aug:
                x1,x2,y=self.flip(x1,x2,y)
                x1,x2,y=self.rotate(x1,x2,y)
                x1,x2,y=self.randomcropscale(x1,x2,y,self.args.bsz,self.args.csz)
                x1,x2,y=self.fixsize(x1,x2,y,self.args.rsz)
                # if self.args.rgb:
                #     img=Image.fromarray(x2.astype('int32'))
                #     tr=RandomColorJitter(brightness_factor=0.3,contrast_factor=0.3,
                #                          saturation_factor=0.3,sharpness_factor=0.3, hue_factor=0.1)
                #     img=tr.call(img)
                #     x2=np.array(img)
                if self.args.denoise:
                    x1=self.gaussianblur(x1)
                if self.args.dehaze:
                    x2=self.dehaze(x2)
                x1, x2 = self.norm(x1, x2)
                x1, x2, y = self.totensor(x1, x2, y)
            else:
                x1, x2, y = self.fixsize(x1, x2, y,self.args.rsz)
                if self.args.denoise:
                    x1 = self.gaussianblur(x1)
                if self.args.dehaze:
                    x2 = self.dehaze(x2)
                x1, x2 = self.norm(x1, x2)
                x1, x2, y = self.totensor(x1, x2, y)
        elif self.split=='val' or self.split=='pre':
            fname3 = self.y[index]
            with rasterio.open(fname3) as patch:
                y = patch.read(list(range(1, 5)))
            y = self.process_label(y)
            y = self.filter_label(x2_tmp, y)
            x1, x2, y = self.fixsize(x1, x2, y, self.args.rsz)
            if self.args.denoise:
                x1 = self.gaussianblur(x1)
            if self.args.dehaze:
                x2 = self.dehaze(x2)
            x1, x2 = self.norm(x1, x2)
            x1, x2, y = self.totensor(x1, x2, y)
        return x1,x2,y,index

    def read_data(self,f1,f2):

        with rasterio.open(f1) as patch:
            x1 = patch.read(list(range(1,3)))

        with rasterio.open(f2) as patch:
            x2 = patch.read(list(range(1,14)))

        return x1,x2

    def process_label(self,y):

        C,H,W=y.shape

        y=y[[0],:,:]#1,h,w, simplified IGBP

        y=y.reshape(-1)

        y = list(map(lambda x: lab_dict[x], y))

        y=np.array(y)#list->array

        y=y.reshape(-1,H,W)

        y-=1 # start from 0

        return y

    def filter_label(self,x2,y):

        x2[np.isnan(x2)] = 0

        x2[x2>10000]=10000
        x2[x2<0]=0

        x2 = x2.astype('float')

        R = x2[:, :, 3]
        G = x2[:, :, 2]
        B = x2[:, :, 1]
        Nir = x2[:, :, 7]  # TM4
        Mir = x2[:, :, 10]  # TM5
        SWir = x2[:, :, 11]  # TM7

        MSI = SWir / Nir
        NDWI = (G - Nir) / (G + Nir)
        NDVI = (Nir - R) / (Nir + R)
        NDBBI = (1.5 * SWir - (Nir + G) / 2.) / (1.5 * SWir + (Nir + G) / 2.)  # å½’ä¸€åŒ–å·®å€¼è£¸åœ°ä¸Žå»ºç­‘ç”¨åœ°æŒ‡æ•°
        #SAVI = (Nir - R) * (1 + 0.5) / (Nir + R + 0.5)  # åœŸå£¤è°ƒæ•´æ¤è¢«æŒ‡æ•°
        #MNDWI = (G - Mir) / (G + Mir)
        BSI = ((Mir + R) - (Nir + B)) / ((Mir + R) + (Nir + B))  # è£¸åœŸæŒ‡æ•°
        NBI = R * SWir / Nir
       # EBSI = (BSI - MNDWI) / (BSI + MNDWI)

        y_clean=y.copy()

        # Forest0,Shrubland1,bg_1-2,Grassland3,Wetlands4,Croplands5,Urban6,bg2-7,Barren8,water9

        if self.args.rule == 'dw_jiu':
            # Forest0,Shrubland1,bg_1-2,Grassland3,Wetlands4,Croplands5,Urban6,bg2-7,Barren8,water9

            # columns

            # 修正不符合要求的森林类
            y_clean[np.where((NDWI < 0.) & (NDVI > 0.75) & (y != 0))] = 10
            # 修正不符合要求的灌木类
            y_clean[np.where((NDWI < 0.) & (NDVI > 0.2) & (NDVI < 0.35) & (MSI > 1.5) & (y != 1))] = 10
            # 修正不符合要求的草地类
            y_clean[np.where((NDWI < 0.) & (NDVI > 0.4) & (NDVI < 0.55) & (y != 3))] = 10
            # 修正不符合要求的湿地类
            y_clean[np.where((NDWI < 0.) & (NDVI > 0.6) & (NDVI < 0.75) & (y != 4))] = 10
            # 修正不符合要求的农田类
            # y_clean[np.where((NDVI > 0.2) & (NDVI < 0.35) & (MSI > 1) & (MSI < 1.5) & (y != 5))] = 10
            # 修正不符合要求的建筑类
            # y_clean[np.where((NDVI > 0.2) & (NDVI < 0.35) & (MSI > 0.9) & (MSI < 1) & (y != 6))] = 10
            # 修正不符合要求的裸地类
            y_clean[np.where((NDWI < 0.) & (NDVI > 0) & (NDVI < 0.15) & (y != 8))] = 10
            # 修正裸土建筑错分到其他类
            # y_clean[np.where((BSI > -0.4) & (NDVI < 0.15) & (y != 6) & (y != 8))] = 10
            # 修正不符合要求的水体类
            y_clean[np.where((NDWI > 0.) & (y != 9))] = 10
            y_clean[np.where((NDWI < 0.) & (y == 9))] = 10
            # rows
            # forest

            # 修正其他类错标到森林
            # shrubland

            # savanna

            # 将热带草原标签修正为森林
            y_clean[np.where((NDWI < 0) & (NDVI > 0.75) & ((y == 30) | (y == 31)))] = 0
            # 将热带草原标签修正为草地
            #y_clean[np.where((NDWI<0) & (NDVI > 0.4) & (NDVI < 0.55) & (y == 2) & (np.sum(NDWI>0) < 1000))] = 3
            # 将热带草原标签修正为湿地
            #y_clean[np.where((NDWI<0) & (NDVI > 0.6) & (NDVI < 0.75) & (y == 2) & (np.sum(y == 9) > 10000))] = 4
            # 将热带草原标签修正为灌木
            #y_clean[np.where((NDWI<0) & (NDVI > 0.2) & (NDVI < 0.35) & (MSI > 1.5) & ((y==30) | (y==31)))] = 1
            # 将热带草原标签修正为裸地
            #y_clean[np.where((NDWI<0) & (NBI > 750) & (NDVI < 0.2) & (NDVI > 0) & ((y==30) | (y==31)))] = 8

            # 将草地标签修正为湿地
            y_clean[np.where((NDWI<0) & (NDVI > 0.6) & (NDVI < 0.75) & (y == 3) & (np.sum(NDWI>0) > 10000))] = 4

            # wetland
            # 将湿地修正为森林
            y_clean[np.where((NDWI<0) & (NDVI > 0.85) & (y == 4))] = 0


        if self.args.rule == 'dw_new':
            # Forest0,Shrubland1,bg_1-2,Grassland3,Wetlands4,Croplands5,Urban6,bg2-7,Barren8,water9

            # columns

            # 修正不符合要求的森林类
            y_clean[np.where((NDWI < 0.) & (NDVI > 0.75) & (y != 0))] = 10
            # 修正不符合要求的灌木类
            y_clean[np.where((NDWI < 0.) & (NDVI > 0.2) & (NDVI < 0.35) & (MSI > 1.5) & (y != 1))] = 10
            # 修正不符合要求的草地类
            y_clean[np.where((NDWI < 0.) & (NDVI > 0.4) & (NDVI < 0.55) & (y != 3))] = 10
            # 修正不符合要求的湿地类
            y_clean[np.where((NDWI < 0.) & (NDVI > 0.6) & (NDVI < 0.75) & (y != 4))] = 10
            # 修正不符合要求的农田类
            # y_clean[np.where((NDVI > 0.2) & (NDVI < 0.35) & (MSI > 1) & (MSI < 1.5) & (y != 5))] = 10
            # 修正不符合要求的建筑类
            # y_clean[np.where((NDVI > 0.2) & (NDVI < 0.35) & (MSI > 0.9) & (MSI < 1) & (y != 6))] = 10
            # 修正不符合要求的裸地类
            y_clean[np.where((NDWI < 0.) & (NDVI > 0) & (NDVI < 0.15) & (y != 8))] = 10
            # 修正裸土建筑错分到其他类
            # y_clean[np.where((BSI > -0.4) & (NDVI < 0.15) & (y != 6) & (y != 8))] = 10
            # 修正不符合要求的水体类
            y_clean[np.where((NDWI > 0.) & (y != 9))] = 10
            y_clean[np.where((NDWI < 0.) & (y == 9))] = 10
            # rows
            # forest

            # 修正其他类错标到森林
            # shrubland

            # savanna

            # 将热带草原标签修正为森林
            y_clean[np.where((NDWI<0) & (NDVI > 0.75) & ((y==30) | (y==31)))]=0
            # 将热带草原标签修正为草地
            y_clean[np.where((NDWI<0) & (NDVI > 0) & (NDVI < 0.75) & (y == 31))] = 3
            # 将热带草原标签修正为湿地
            #y_clean[np.where((NDWI<0) & (NDVI > 0.6) & (NDVI < 0.75) & (y == 2) & (np.sum(y == 9) > 10000))] = 4
            # 将热带草原标签修正为灌木
            y_clean[np.where((NDWI<0) & (NDVI > 0) & (NDVI < 0.75) & (y == 30))] = 1
            # 将热带草原标签修正为裸地
            #y_clean[np.where((NDWI<0) & (NBI > 750) & (NDVI < 0.2) & (NDVI > 0) & ((y==30) | (y==31)))] = 8

            # 将草地标签修正为湿地
            y_clean[np.where((NDWI<0) & (NDVI > 0.6) & (NDVI < 0.75) & (y == 3) & (np.sum(NDWI>0) > 10000))] = 4

            # wetland
            # 将湿地修正为森林
            y_clean[np.where((NDWI<0) & (NDVI > 0.85) & (y == 4))] = 0

        if self.split!='pre':

            y_clean[y_clean == 30] = 10
            y_clean[y_clean == 31] = 10
            y_clean[y_clean==7] = 10

        if self.split=='pre':
            y_clean[y_clean == 30] = 2
            y_clean[y_clean == 31] = 2

        return y_clean

    def clean(self,x1,x2):
        # s1
        x1[np.isnan(x1)] = 0
        # s2
        x2[np.isnan(x2)] = 0

        # s1_recommend
        x1[x1<-25]=-25
        x1[x1>0]=0
        # s2_recommend
        x2[x2 < 0] = 0
        x2[x2>10000]=10000

        return x1.astype('float32'),x2.astype('float32')
    def flip(self,x1,x2,y):
        p1 = np.random.random()
        if p1<0.5:
            x1 = x1[::-1, :, :]
            x2 = x2[::-1, :, :]
            y = y[:, ::-1, :]
        p2 = np.random.random()
        if p2<0.5:
            x1 = x1[:, ::-1, :]
            x2 = x2[:, ::-1, :]
            y = y[:, :, ::-1]
        return x1,x2,y
    def rotate(self,x1,x2,y):
        p=np.random.random()
        if p < 0.25:
            x1 = np.rot90(x1, 1, (0, 1)).copy()
            x2 = np.rot90(x2, 1, (0, 1)).copy()
            y = np.rot90(y, 1, (1, 2)).copy()
        elif p >= 0.25 and p < 0.5:
            x1 = np.rot90(x1, 2, (0, 1)).copy()
            x2 = np.rot90(x2, 2, (0, 1)).copy()
            y = np.rot90(y, 2, (1, 2)).copy()
        elif p >= 0.5 and p < 0.75:
            x1 = np.rot90(x1, 3, (0, 1)).copy()
            x2 = np.rot90(x2, 3, (0, 1)).copy()
            y = np.rot90(y, 3, (1, 2)).copy()
        return x1,x2, y

    def randomcropscale(self,x1,x2,y,base_size,crop_size):
        # random scale (short edge)
        short_size = np.random.randint(int(base_size * 0.9), int(base_size * 1.1))
        h, w, _ = x1.shape
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        xx1 = np.zeros([ow, oh, x1.shape[-1]])
        xx2 = np.zeros([ow, oh, x2.shape[-1]])
        yy = np.zeros([y.shape[0], ow, oh])
        for i in range(x1.shape[-1]):
            xx1[:, :, i] = cv2.resize(x1[:, :, i], (ow, oh), interpolation=cv2.INTER_LINEAR)
        for i in range(x2.shape[-1]):
            xx2[:, :, i] = cv2.resize(x2[:, :, i], (ow, oh), interpolation=cv2.INTER_LINEAR)
        for i in range(y.shape[0]):
            yy[i, :, :] = cv2.resize(y[i, :, :], (ow, oh), interpolation=cv2.INTER_NEAREST)
        # random crop crop_size, default:short_size>crop_size
        h, w, _ = xx1.shape
        m1 = np.random.randint(0, h - crop_size)
        n1 = np.random.randint(0, w - crop_size)
        x1 = xx1[m1:m1 + crop_size, n1:n1 + crop_size, :]
        x2 = xx2[m1:m1 + crop_size, n1:n1 + crop_size, :]
        y = yy[:, m1:m1 + crop_size, n1:n1 + crop_size]
        return x1,x2, y
    def fixsize(self,x1,x2,y,rsz):
        xx1 = np.zeros([rsz, rsz, x1.shape[-1]])
        xx2 = np.zeros([rsz, rsz, x2.shape[-1]])
        yy = np.zeros([y.shape[0], rsz, rsz])
        for i in range(x1.shape[-1]):
            xx1[:, :, i] = cv2.resize(x1[:, :, i], (rsz, rsz), interpolation=cv2.INTER_LINEAR)
        for i in range(x2.shape[-1]):
            xx2[:, :, i] = cv2.resize(x2[:, :, i], (rsz, rsz), interpolation=cv2.INTER_LINEAR)
        for i in range(y.shape[0]):
            yy[i, :, :] = cv2.resize(y[i, :, :], (rsz, rsz), interpolation=cv2.INTER_NEAREST)
        return xx1,xx2,yy

    def gaussianblur(self,x1,sigma=1.5):
        for i in range(x1.shape[-1]):
            x1[:,:,i]=gaussian_filter(x1[:,:,i],sigma)
        return x1

    def dehaze(self, x, r=9, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
        Y = np.zeros(x.shape)
        V1, A = getV1(x, r, eps, w, maxV1)  # å¾—åˆ°é®ç½©å›¾åƒå’Œå¤§æ°”å…‰ç…§
        for k in range(x.shape[2]):
            Y[:, :, k] = (x[:, :, k] - V1) / (1 - V1 / A)  # é¢œè‰²æ ¡æ­£

        if bGamma:
            Y = Y ** (np.log(0.5) / np.log(Y.mean()))  # gammaæ ¡æ­£,é»˜è®¤ä¸è¿›è¡Œè¯¥æ“ä½œ
        return Y
    def norm(self,x1,x2):
        # input,x1:[-25,0],x2:[0,5000]
        h,w,c1=x1.shape
        h,w,c2=x2.shape
        x2 /= 10000 * 1.0
        if self.args.scale == 'std':
            x1 = x1.reshape(-1, c1)
            x1 -= self.s1_mean
            x1 /= self.s1_std
        if self.args.scale == 'norm':
            # recommend: x1[0,1],x2:[0,1]
            x1 = (x1 + 25) / 25 * 1.0

        if self.args.rgb :
            x2 -= MEAN
            x2 /= STD
        else:#RS image
            if self.args.scale == 'std':
                x2=x2.reshape(-1,c2)
                x2-=self.s2_mean
                x2/=self.s2_std

        return x1.reshape(h,w,c1),x2.reshape(h,w,c2)
    def totensor(self,x1,x2,y):
        x1 = torch.from_numpy(x1.transpose(2, 0, 1)).float()
        x2 = torch.from_numpy(x2.transpose(2, 0, 1)).float()
        y=torch.from_numpy(y).float()
        return x1,x2,y

class RandomColorJitter(object):
    def __init__(self, brightness_factor,contrast_factor, saturation_factor,
                 sharpness_factor, hue_factor):
        self.bri_factor=brightness_factor
        self.con_factor=contrast_factor
        self.sat_factor=saturation_factor
        self.sha_factor=sharpness_factor
        self.hue_factor=hue_factor
    def call(self,x):
        img = x

        bri_p = random.uniform(1 - self.bri_factor, 1 + self.bri_factor)
        con_p = random.uniform(1 - self.con_factor, 1 + self.con_factor)
        sat_p = random.uniform(1 - self.sat_factor, 1 + self.sat_factor)
        sha_p = random.uniform(1 - self.sha_factor, 1 + self.sha_factor)
        hue_p = random.uniform(0 - self.hue_factor, 0 + self.hue_factor)

        img = ImageEnhance.Brightness(img).enhance(bri_p)
        img = ImageEnhance.Contrast(img).enhance(con_p)
        img = ImageEnhance.Color(img).enhance(sat_p)
        img = ImageEnhance.Sharpness(img).enhance(sha_p)

        input_mode = img.mode

        if input_mode in {'L', '1', 'I', 'F'}:
            return img
        else:
            h, s, v = img.convert('HSV').split()

            np_h = np.array(h, dtype=np.uint32)
            # uint8 addition take cares of rotation across boundaries
            with np.errstate(over='ignore'):
                np_h += np.uint32(hue_p * 255)
            h = Image.fromarray(np_h, 'L')

            img = Image.merge('HSV', (h, s, v)).convert(input_mode)

            return img