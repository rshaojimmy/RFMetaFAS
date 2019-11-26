import itertools
import os
from collections import OrderedDict
import torchvision.utils as vutils
import torch
import torch.optim as optim
from torch import nn
from misc.utils import get_inf_iterator, mkdir, mixup_process_intra, mixup_process_cross
from misc import evaluate
from torch.nn import DataParallel
import random
import torch.autograd as autograd
from copy import deepcopy
from itertools import permutations, combinations
import models

from pdb import set_trace as st

def Train(args, FeatExtor, DepthEstor, FeatEmbder,
        data_loader1_real, data_loader1_fake,
        data_loader2_real, data_loader2_fake,
        data_loader3_real, data_loader3_fake,
        data_loader_target,
        summary_writer, Saver, savefilename):
            
    ####################
    # 1. setup network #
    ####################
    # set train state for Dropout and BN layers
    FeatExtor.train()
    DepthEstor.train()
    FeatEmbder.train()


    FeatExtor = DataParallel(FeatExtor)    
    DepthEstor = DataParallel(DepthEstor)    
 


    # setup criterion and optimizer
    criterionCls = nn.BCEWithLogitsLoss()
    criterionDepth = torch.nn.MSELoss()


    if args.optimizer_meta is 'adam':

        optimizer_all = optim.Adam(itertools.chain(FeatExtor.parameters(), DepthEstor.parameters(), FeatEmbder.parameters()),
                                   lr=args.lr_meta,
                                   betas=(args.beta1, args.beta2))

    else:
        raise NotImplementedError('Not a suitable optimizer')
    


    iternum = max(len(data_loader1_real),len(data_loader1_fake),
                  len(data_loader2_real),len(data_loader2_fake), 
                  len(data_loader3_real),len(data_loader3_fake))        

    print('iternum={}'.format(iternum))

    ####################
    # 2. train network #
    ####################
    global_step = 0

    for epoch in range(args.epochs):

        data1_real = get_inf_iterator(data_loader1_real)
        data1_fake = get_inf_iterator(data_loader1_fake)

        data2_real = get_inf_iterator(data_loader2_real)
        data2_fake = get_inf_iterator(data_loader2_fake)

        data3_real = get_inf_iterator(data_loader3_real)
        data3_fake = get_inf_iterator(data_loader3_fake)
            

        for step in range(iternum):

            #============ one batch extraction ============#

            cat_img1_real, depth_img1_real, lab1_real = next(data1_real)
            cat_img1_fake, depth_img1_fake, lab1_fake = next(data1_fake)

            cat_img2_real, depth_img2_real, lab2_real = next(data2_real)
            cat_img2_fake, depth_img2_fake, lab2_fake = next(data2_fake)

            cat_img3_real, depth_img3_real, lab3_real = next(data3_real)
            cat_img3_fake, depth_img3_fake, lab3_fake = next(data3_fake)

            #============ one batch collection ============# 

            catimg1 = torch.cat([cat_img1_real,cat_img1_fake],0).cuda()
            depth_img1 = torch.cat([depth_img1_real,depth_img1_fake],0).cuda()
            lab1 = torch.cat([lab1_real,lab1_fake],0).float().cuda()

            catimg2 = torch.cat([cat_img2_real,cat_img2_fake],0).cuda()
            depth_img2 = torch.cat([depth_img2_real,depth_img2_fake],0).cuda()
            lab2 = torch.cat([lab2_real,lab2_fake],0).float().cuda()

            catimg3 = torch.cat([cat_img3_real,cat_img3_fake],0).cuda()
            depth_img3 = torch.cat([depth_img3_real,depth_img3_fake],0).cuda()
            lab3 = torch.cat([lab3_real,lab3_fake],0).float().cuda()

            catimg = torch.cat([catimg1,catimg2,catimg3],0)
            depth_GT = torch.cat([depth_img1,depth_img2,depth_img3],0)
            label = torch.cat([lab1,lab2,lab3],0)

           #============ doamin list augmentation ============# 
            catimglist = [catimg1,catimg2,catimg3]
            lablist = [lab1,lab2,lab3]
            deplist = [depth_img1,depth_img2,depth_img3]

            domain_list = list(range(len(catimglist)))
            random.shuffle(domain_list) 
            
            meta_train_list = domain_list[:args.metatrainsize] 
            meta_test_list = domain_list[args.metatrainsize:]
            print('metatrn={}, metatst={}'.format(meta_train_list, meta_test_list[0]))
 
            
            #============ meta training ============#

            Loss_dep_train = 0.0
            Loss_cls_train = 0.0

            adapted_state_dicts = []

            for index in meta_train_list:

                catimg_meta = catimglist[index]
                lab_meta = lablist[index]
                depGT_meta = deplist[index]

                batchidx = list(range(len(catimg_meta)))
                random.shuffle(batchidx)
                
                img_rand = catimg_meta[batchidx,:]
                lab_rand = lab_meta[batchidx]
                depGT_rand = depGT_meta[batchidx,:]

                feat_ext_all,feat = FeatExtor(img_rand)
                pred = FeatEmbder(feat)
                depth_Pre = DepthEstor(feat_ext_all)

                Loss_cls = criterionCls(pred.squeeze(), lab_rand)
                Loss_dep = criterionDepth(depth_Pre, depGT_rand)

                Loss_dep_train+=Loss_dep
                Loss_cls_train+=Loss_cls

                zero_param_grad(FeatEmbder.parameters())    
                grads_FeatEmbder = torch.autograd.grad(Loss_cls, FeatEmbder.parameters(), create_graph=True)
                fast_weights_FeatEmbder = FeatEmbder.cloned_state_dict()

                adapted_params = OrderedDict()
                for (key, val), grad in zip(FeatEmbder.named_parameters(), grads_FeatEmbder):
                    adapted_params[key] = val - args.meta_step_size * grad
                    fast_weights_FeatEmbder[key] = adapted_params[key]   

                adapted_state_dicts.append(fast_weights_FeatEmbder)


            #============ meta testing ============#    
            Loss_dep_test = 0.0
            Loss_cls_test = 0.0

            index = meta_test_list[0]

            catimg_meta = catimglist[index]
            lab_meta = lablist[index]
            depGT_meta = deplist[index]

            batchidx = list(range(len(catimg_meta)))
            random.shuffle(batchidx)
            
            img_rand = catimg_meta[batchidx,:]
            lab_rand = lab_meta[batchidx]
            depGT_rand = depGT_meta[batchidx,:]

            feat_ext_all,feat = FeatExtor(img_rand)
            depth_Pre = DepthEstor(feat_ext_all)
            Loss_dep = criterionDepth(depth_Pre, depGT_rand)

            for n_scr in range(len(meta_train_list)):
                a_dict = adapted_state_dicts[n_scr]

                pred = FeatEmbder(feat, a_dict)
                Loss_cls = criterionCls(pred.squeeze(), lab_rand)

                Loss_cls_test+=Loss_cls

            Loss_dep_test = Loss_dep

            Loss_dep_train_ave = Loss_dep_train/len(meta_train_list)   
            Loss_dep_test = Loss_dep_test  

            Loss_meta_train =  Loss_cls_train+args.W_depth*Loss_dep_train  
            Loss_meta_test =  Loss_cls_test+args.W_depth*Loss_dep_test                 

            Loss_all = Loss_meta_train + args.W_metatest*Loss_meta_test

            optimizer_all.zero_grad()
            Loss_all.backward()
            optimizer_all.step()        
                                   


            if (step+1) % args.log_step == 0:
                errors = OrderedDict([
                                    ('Loss_meta_train', Loss_meta_train.item()),
                                    ('Loss_meta_test', Loss_meta_test.item()),
                                    ('Loss_cls_train', Loss_cls_train.item()),
                                    ('Loss_cls_test', Loss_cls_test.item()),
                                    ('Loss_dep_train_ave', Loss_dep_train_ave.item()),
                                    ('Loss_dep_test', Loss_dep_test.item()),
                                    ])
                Saver.print_current_errors((epoch+1), (step+1), errors)


            #============ tensorboard the log info ============#
            info = {
                'Loss_meta_train': Loss_meta_train.item(),                                                                                     
                'Loss_meta_test': Loss_meta_test.item(),                                                                                     
                'Loss_cls_train': Loss_cls_train.item(),  
                'Loss_cls_test': Loss_cls_test.item(),                                                                                                                                                                                                                                                          
                'Loss_dep_train_ave': Loss_dep_train_ave.item(),                                                                                                                                                                                                                                                          
                'Loss_dep_test': Loss_dep_test.item(),                                                                                                                                                                                                                                                          
                    }           
            for tag, value in info.items():
                summary_writer.add_scalar(tag, value, global_step) 


            global_step+=1


            #############################
            # 2.4 save model parameters #
            #############################
            if ((step + 1) % args.model_save_step == 0):
                model_save_path = os.path.join(args.results_path, 'snapshots', savefilename)     
                mkdir(model_save_path) 

                torch.save(FeatExtor.state_dict(), os.path.join(model_save_path,
                    "FeatExtor-{}-{}.pt".format(epoch+1, step+1)))
                torch.save(FeatEmbder.state_dict(), os.path.join(model_save_path,
                    "FeatEmbder-{}-{}.pt".format(epoch+1, step+1)))
                torch.save(DepthEstor.state_dict(), os.path.join(model_save_path,
                    "DepthEstor-{}-{}.pt".format(epoch+1, step+1)))


        if ((epoch + 1) % args.model_save_epoch == 0):
            model_save_path = os.path.join(args.results_path, 'snapshots', savefilename)     
            mkdir(model_save_path) 

            torch.save(FeatExtor.state_dict(), os.path.join(model_save_path,
                "FeatExtor-{}.pt".format(epoch+1)))
            torch.save(FeatEmbder.state_dict(), os.path.join(model_save_path,
                "FeatEmbder-{}.pt".format(epoch+1)))
            torch.save(DepthEstor.state_dict(), os.path.join(model_save_path,
                "DepthEstor-{}.pt".format(epoch+1)))


    torch.save(FeatExtor.state_dict(), os.path.join(model_save_path,
        "FeatExtor-final.pt"))
    torch.save(FeatEmbder.state_dict(), os.path.join(model_save_path,
        "FeatEmbder-final.pt"))    
    torch.save(DepthEstor.state_dict(), os.path.join(model_save_path,
        "DepthEstor-final.pt"))


def zero_param_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()
