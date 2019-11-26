import numpy as np
import os
import ntpath
import time
from . import utils

from pdb import set_trace as st

class Saver():
    def __init__(self, args, logfilename):

        self.args = args

        self.save_file = os.path.join(args.results_path, 'log', logfilename)
        if not os.path.exists(self.save_file):
            utils.mkdirs(self.save_file) 

        self.log_name = os.path.join(self.save_file, 'loss_log.txt')

        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)


        self.imgsave_dir = os.path.join(self.save_file, 'images')
        if not os.path.exists(self.imgsave_dir):
        # print('create image directory %s...' % self.imgsave_dir)
            utils.mkdirs(self.imgsave_dir)


    def print_current_errors(self, epoch, i, errors):
        message = '(epoch: %d, iters: %d) ' % (epoch, i)
        for k, v in errors.items():
            message += '%s: %.5f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    #save image to the disk
    # def save_images(self, visuals, image_path, epoch):
    #     image_dir = self.imgsave_dir
    #     short_path = ntpath.basename(image_path[0])
    #     name = os.path.splitext(short_path)[0]

    #     for label, image_numpy in visuals.items():
    #         image_name = '%d_%s_%s.png' % (epoch, label, name)
    #         save_path = os.path.join(image_dir, image_name)
    #         utils.save_image(image_numpy, save_path)

    def save_images(self, image_numpy, epoch, step, imgidx, lab):
        image_dir_root = self.imgsave_dir
        image_dir = os.path.join(image_dir_root, 'epoch'+str(epoch)+ '_'+'step'+str(step))
        if not os.path.exists(image_dir):
            utils.mkdirs(image_dir) 

        image_name = '%s-%d_%s-%d.png' % ('imgidx', imgidx, 'label', lab)
        save_path = os.path.join(image_dir, image_name)
        utils.save_image(image_numpy, save_path)


        # save to the disk
    def print_config(self):
        opt = vars(self.args)
        file_name = os.path.join(self.save_file, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(opt.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

       

