import cv2
import os 
import random
import logging
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
from os import listdir
from os.path import splitext
from glob import glob
from PIL import Image

def augmix(image, preprocess=transforms.ToTensor()):
    aug_list = augmentations.augmentations

    ws = np.float32(np.random.dirichlet([1] * 3))
    m = np.float32(np.random.beta(1, 1))

    mix = torch.zeros_like(preprocess(image))
    for i in range(3):
        image_aug = image.copy()
        op = np.random.choice(aug_list)
        image_aug = op(image_aug, 3)
        mix += ws[i] * preprocess(image_aug)
    mixed = (1 - m) * preprocess(image) + m * mix
    return mixed

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, sources=[], val=False, full=False):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.val = val
        self.full = full
        self.sources = sources

        imgs_all = os.listdir(imgs_dir)
        self.ids = []
        if len(sources) != 0:
            tmp = []
            for c, source in enumerate(sources):
                curr_imgs = [i for i in imgs_all if source in i]
                for curr_img in curr_imgs:
                    tmp.append(curr_img)
            self.ids += tmp 
        else:
            self.ids = imgs_all
        
        if not full:
            total_num = len(self.ids)
            num_train = int(total_num * 0.8)
            random.seed(0)
            random.shuffle(self.ids)
            if val:
                self.ids = self.ids[num_train:]
            else:
                self.ids = self.ids[:num_train]
        print(len(self.ids))

        transform_list = [transforms.Resize((128, 128), 0),
                          transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file  = os.path.join(self.imgs_dir, idx)
        mask_file = os.path.join(self.masks_dir, idx)
        
        #print(img_file)
        mask = Image.open(mask_file)
        mask = self.transform(mask)

        #img = Image.open(img_file[0])
        img = Image.open(img_file).convert("RGB")

        image = self.transform(img)
        image_min, image_max = image.min(), image.max()
        image = (image - image.min()) / (image.max() - image.min())
        
        return {
            'image': image,
            'mask': mask,
            'path': img_file,
        }


class GMDataset(Dataset):
    def __init__(self, imgs_dir, mask_dir, sources=[], phase='test', val=False, full=False, select=None):
        self.imgs_dir = imgs_dir
        self.mask_dir = mask_dir
        self.sources = sources

        all_images = os.listdir(self.imgs_dir)
        self.image_ids = []
        for source in sources:
            self.image_ids += [name for name in all_images if source in name]
        self.mask_ids = os.listdir(self.mask_dir)
        self.phase = phase

        transform_list = [transforms.CenterCrop((144,144)),
                          transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)
        
        if not full:
            total_num = len(self.image_ids)
            num_train = int(total_num * 0.8)
            random.seed(0)
            random.shuffle(self.image_ids)
            if val:
                self.image_ids = self.image_ids[num_train:]
            else:
                self.image_ids = self.image_ids[:num_train]
        
        if select is not None:
            self.image_ids = [id for id in self.image_ids if id in select]
        print('dataset size %s' % len(self.image_ids))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, i):
        curr_image_name = self.image_ids[i]
        curr_image_ori = Image.open(os.path.join(self.imgs_dir, curr_image_name)).convert("L")

        image = self.transform(curr_image_ori)
        image_min, image_max = image.min(), image.max()
        image = (image - image.min()) / (image.max() - image.min())

        tmp = curr_image_name.split('-')
        tmp[2] = 'mask'
        tmp.append(tmp[-1])
        tmp[3] = 'sp'
        curr_mask_name = ''
        for t in tmp:
            curr_mask_name += t + '-'
        curr_mask_name = curr_mask_name[:-1]

        curr_mask = Image.open(os.path.join(self.mask_dir, curr_mask_name)).convert("L")
        curr_mask = self.transform(curr_mask)
        
        if curr_mask.shape[0] != 1:
            curr_mask = curr_mask[0].unsqueeze(0)
        
        return {
            'image': image,
            'mask': curr_mask,
            'path': curr_image_name, 
        }


class RetinaDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, cat='hrf', full=False, val=False, select=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        
        self.image_paths = []
        for c in cat:
            self.image_paths += [im for im in os.listdir(imgs_dir) if c in im]
        self.transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])

        if not full:
            random.seed(0)
            random.shuffle(self.image_paths)
            num = len(self.image_paths)
            split = int(num * 0.8)
            if val:
                self.image_paths = self.image_paths[split:]
            else:
                self.image_paths = self.image_paths[:split]

        if select is not None:
            self.image_paths = [id for id in self.image_paths if id in select]
        print(self.image_paths)
        print('length of dataset', len(self.image_paths))
                
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_name = image_path.split('.')[0]
        mask_path = image_name + '.png' 

        curr_image_ori = Image.open(os.path.join(self.imgs_dir, image_path))
        image = self.transform(curr_image_ori)
        image_min, image_max = image.min(), image.max()
        image = (image - image.min()) / (image.max() - image.min())

        mask = Image.open(os.path.join(self.masks_dir, mask_path))
        mask = self.transform(mask)

        return {
            'image': image,
            'mask': mask,
            'path': image_path,
        }
