import os
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
# import matplotlib.pyplot as plt
from PIL import Image
from misc import utils
from pdb import set_trace as st

def default_loader(path):
    RGBimg = Image.open(path).convert('RGB')
    HSVimg = Image.open(path).convert('HSV')
    return RGBimg, HSVimg


class DatasetLoader(Dataset):
    def __init__(self, name, transform=None, loader=default_loader, root='../../../datasets/'):

        self.name = name
        self.root = os.path.expanduser(root)
        self.root = os.path.join(self.root, self.name)
        filename = 'image_list_all.txt'

        fh = open(os.path.join(self.root, filename), 'r')

        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        fn = os.path.join(self.root, fn)
        rgbimg, hsvimg = self.loader(fn)
        if self.transform is not None:
            rgbimg = self.transform(rgbimg)
            hsvimg = self.transform(hsvimg)

            catimg = torch.cat([rgbimg,hsvimg],0)

        return catimg, label

    def __len__(self):
        return len(self.imgs)


def get_tgtdataset_loader(name, batch_size):

    # pre_process = transforms.Compose([transforms.ToTensor(),
    #                                   transforms.Normalize(
    #                                   mean=[0.485, 0.456, 0.406],
    #                                   std=[0.229, 0.224, 0.225])])      

    pre_process = transforms.Compose([transforms.ToTensor()])  


    # dataset and data loader
    dataset = DatasetLoader(name=name,
                        transform=pre_process
                        )

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True)

    return data_loader
