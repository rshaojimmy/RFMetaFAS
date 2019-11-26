"""Test script for ATDA."""

import torch.nn as nn
import torch
import numpy as np

from misc.utils import make_variable, tensor2im
from pdb import set_trace as st

def evaluate(FeatExtor, FeatEmbder, Classifier, data_loader):
    """Evaluate classifier on source or target domains."""
    # set eval state for Dropout and BN layers
    FeatExtor.eval()
    FeatEmbder.eval()
    Classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    testnum = 0
    for (rgbimages, hsvimages, labels) in data_loader:

        images = torch.cat([rgbimages,hsvimages],1)
        images = images.cuda()
        labels = labels.long().squeeze().cuda()

        feat_ext,_,_= FeatExtor(images)
        feat_embd = FeatEmbder(feat_ext)
        preds = Classifier(feat_embd)

        loss += criterion(preds, labels).item()

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum().item()

        testnum +=1
        if testnum == 50:
            break

    loss /= (testnum)
    acc /= (testnum*images.size()[0])

    # loss /= len(data_loader)
    # acc /= len(data_loader.dataset)

    print("Avg Loss = {:.5f}, Avg Accuracy = {:2.5%}".format(loss, acc))



def evaluate_img(FeatExtor, DepthEsmator, data_loader, epoch, step, Saver):
    """Evaluate classifier on source or target domains."""
    # set eval state for Dropout and BN layers
    FeatExtor.eval()
    DepthEsmator.eval()


    # evaluate network
    testnum = 0
    imgidx = 0
    for (catimages, labels) in data_loader:

        images = catimages.cuda()
        labels = labels.long().squeeze().cuda()

        feat_ext,_= FeatExtor(images)
        depthimg = DepthEsmator(feat_ext)

        for i in range(depthimg.size()[0]):
            depthimg_npy = tensor2im(depthimg[i])
            lab = labels[i].item()
            Saver.save_images(depthimg_npy, epoch, step, imgidx, lab)

            imgidx+=1

        testnum +=1
        if testnum == 1:
            break





