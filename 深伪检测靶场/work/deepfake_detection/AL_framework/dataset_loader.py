import torch
from PIL import Image 
import os, json, glob
import cv2
import pandas as pd
import random
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, root, transform=None, num_classes=2):
        super(BaseDataset,self).__init__()
        self.root = root
        self.transform = transform
        self.num_classes = num_classes
        assert transform is not None, "transform is None"

    def __getitem__(self,idx):
        img_path = self.imgs[idx][0]
        label = self.imgs[idx][1]

        image = Image.open(img_path).convert('RGB')

        image = self.transform(image)

        return image, label, img_path

    def __len__(self):
        return len(self.imgs)




class train_loader(BaseDataset):
    def __init__(self,root,transform=None, test=False):
        super(train_loader,self).__init__(root=root, transform=transform)

        real_root = os.path.join(root, "real")
        synthesis_root = os.path.join(root, "fake")

        real_imgs = []
        fake_imgs = []

        real_imgs = glob.glob(os.path.join(real_root, "*"))
        fake_imgs = glob.glob(os.path.join(synthesis_root, "*"))

        print("fake imgs count :{}, real imgs count :{}".format(len(fake_imgs),len(real_imgs)))
        fake_imgs = [[p,1] for p in fake_imgs]
        real_imgs = [[p,0] for p in real_imgs]
        self.imgs = fake_imgs + real_imgs

        random.shuffle(self.imgs)

        # if test:
        #     total_len = len(self.imgs)
        #     self.imgs = self.imgs[:int(total_len*0.1)]



class query_loader(BaseDataset):
    def __init__(self,root,transform=None):
        super(train_loader,self).__init__(root=root, transform=transform)

        real_imgs = []

        real_imgs = glob.glob(os.path.join(root, "*"))

        print("imgs count :{}".format(len(real_imgs)))

        real_imgs = [[p,0] for p in real_imgs]
        self.imgs = real_imgs

        random.shuffle(self.imgs)



