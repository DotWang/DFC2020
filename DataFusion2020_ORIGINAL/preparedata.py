import numpy as np
from tqdm import tqdm
from sen12ms_dataLoader import *

def load_data(args):
    # s1_trn=[]
    # s2_trn=[]
    # y_trn=[]
    # s1_val=[]
    # s2_val=[]
    # y_val=[]

    print('Loading data...')

    sen12ms = SEN12MSDataset("/data/PublicData/DF2020/trn/")

    IDs = np.load('/data/PublicData/DF2020/trn/clean.npy')

    N = IDs.shape[0]

    # for i in tqdm([Seasons.SPRING,Seasons.SUMMER,Seasons.FALL,Seasons.WINTER]):
    #
    #     s1_data, s2_data, y_data = sen12ms.get_triplets(i, s1_bands=S1Bands.ALL,
    #                                       s2_bands=S2Bands.ALL, lc_bands=LCBands.ALL)


        #N=len(s1_data)

    #s1_data=IDs
    # s2_data=IDs
    # y_data=IDs

    #s1_data=np.array(s1_data)
    # s2_data=np.array(s2_data)
    # y_data=np.array(y_data)

    idx=np.arange(N)
    np.random.shuffle(idx)

    trn_ids=IDs[idx[:int(N * args.trn_ratio)],:]
    val_ids=IDs[idx[int(N * args.trn_ratio):int(N * (args.trn_ratio + args.val_ratio))],:]

    # s1_trn.extend(s1_data[idx[:int(N * args.trn_ratio)]])
    # s1_val.extend(s1_data[idx[int(N * args.trn_ratio):int(N * (args.trn_ratio + args.val_ratio))]])

    # s2_trn.extend(s2_data[idx[:int(N * args.trn_ratio)]])
    # s2_val.extend(s2_data[idx[int(N * args.trn_ratio):int(N * (args.trn_ratio + args.val_ratio))]])
    #
    # y_trn.extend(y_data[idx[:int(N * args.trn_ratio)]])
    # y_val.extend(y_data[idx[int(N * args.trn_ratio):int(N * (args.trn_ratio + args.val_ratio))]])

    #s1_trn=np.stack(s1_trn,axis=0)
    # s2_trn=np.stack(s2_trn,axis=0)
    # y_trn=np.stack(y_trn,axis=0)

    #s1_val=np.stack(s1_val,axis=0)
    # s2_val=np.stack(s2_val,axis=0)
    # y_val=np.stack(y_val,axis=0)

    s1_trn_name=[]
    s2_trn_name=[]
    y_trn_name=[]

    s1_val_name = []
    s2_val_name = []
    y_val_name = []

    season_dict={1:Seasons.SPRING,2:Seasons.SUMMER,3:Seasons.FALL,4:Seasons.WINTER}

    print('loading training files...')

    for i in tqdm(range(trn_ids.shape[0])):
        s1_name,s2_name,y_name=sen12ms.get_s1s2lc_triplet(season_dict[trn_ids[i,0]], trn_ids[i,1], trn_ids[i,2],
                                                                               s1_bands=S1Bands.ALL,s2_bands=S2Bands.ALL, lc_bands=LCBands.ALL)
        s1_trn_name.append(s1_name)
        s2_trn_name.append(s2_name)
        y_trn_name.append(y_name)

    print('loading valing files...')

    for i in tqdm(range(val_ids.shape[0])):
        s1_name,s2_name,y_name=sen12ms.get_s1s2lc_triplet(season_dict[val_ids[i,0]], val_ids[i,1], val_ids[i,2],
                                                                               s1_bands=S1Bands.ALL,s2_bands=S2Bands.ALL, lc_bands=LCBands.ALL)
        s1_val_name.append(s1_name)
        s2_val_name.append(s2_name)
        y_val_name.append(y_name)

    s1_trn_name = np.array(s1_trn_name)
    s2_trn_name = np.array(s2_trn_name)
    y_trn_name = np.array(y_trn_name)
    
    s1_val_name = np.array(s1_val_name)
    s2_val_name = np.array(s2_val_name)
    y_val_name = np.array(y_val_name)

    return s1_trn_name,s2_trn_name,y_trn_name,s1_val_name,s2_val_name,y_val_name
