"""Adversarial adaptation to train target encoder."""
import os
from collections import OrderedDict
import torchvision.utils as vutils
import torch
import torch.optim as optim
from torch import nn
from misc.utils import get_inf_iterator, mkdir
from misc import evaluate
from torch.nn import DataParallel
import numpy as np
import h5py
import torch.nn.functional as F


from pdb import set_trace as st


def Test(args, FeatExtor, FeatEmbder, 
       data_loader_target,
       savefilename):

    # print("***The type of norm is: {}".format(normtype))

    savepath = os.path.join(args.results_path, savefilename)
    mkdir(savepath)  
    ####################
    # 1. setup network #
    ####################
    # set train state for Dropout and BN layers
    FeatExtor.eval()
    FeatEmbder.eval()

    # FeatEmbder = DataParallel(FeatEmbder)    
    # FeatExtor = DataParallel(FeatExtor)    

    score_list = []
    label_list = []

    idx = 0

    for (catimages, labels) in data_loader_target:

        images = catimages.cuda()
        # labels = labels.long().squeeze().cuda()

        _,feat  = FeatExtor(images)
        label_pred  = FeatEmbder(feat)
        
        score = torch.sigmoid(label_pred).cpu().detach().numpy()
        labels = labels.numpy()

        score_list.append(score.squeeze())
        label_list.append(labels)

        print('SampleNum:{} in total:{}, score:{}'.format(idx,len(data_loader_target), score.squeeze()))

        idx+=1

    with h5py.File(os.path.join(savepath, 'Test_data.h5'), 'w') as hf:
        hf.create_dataset('score', data=score_list)
        hf.create_dataset('label', data=label_list)


   




