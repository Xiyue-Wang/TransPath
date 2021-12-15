import torch
from PIL import Image
import json
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import numpy as np
import cv2
image_folder=r'/all_image_path'
all_file=os.listdir(path)

class tcgaataset_byol(Dataset):
    def __init__(self,
                 df,
                 transform=None,
                 ):
        image_size = 256
        p_blur = 0.5
        self.image_folder = image_folder
        # self.label=label
        self.df = df
        self.transform1 = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            T.RandomApply([T.GaussianBlur(kernel_size=image_size // 20 * 2 + 1, sigma=(0.1, 2.0))], p=p_blur),
            T.RandomGrayscale(p=0.2),


            T.ToTensor(),

            T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),

        ])
        self.transform2 = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            T.RandomApply([T.GaussianBlur(kernel_size=image_size // 20 * 2 + 1, sigma=(0.1, 2.0))], p=p_blur),
            T.RandomGrayscale(p=0.2),


            T.ToTensor(),

            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),

        ])


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_id = self.df[index]
        img = Image.open(img_id).convert('RGB')
        assert isinstance(img, Image.Image), \
            'The output from the data source must be an Image, got: {}. \
            Please ensure that the list file does not contain labels.'.format(
            type(img))

        q = self.transform1(img)
        k = self.transform2(img)
        return q,k



train_tcga_byol = tcgaataset_byol(all_file)
