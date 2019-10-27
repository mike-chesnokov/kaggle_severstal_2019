# Datasets for Segmentation and Multilable Classification
import os
import random

import cv2
import numpy as np
import pandas as pd
import jpeg4py as jpeg
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import albumentations as albu

from utils import (
    rle2mask,
    _fix_seeds
)
from configs import SEED


# Datasets for segmentation

class SteelTrainDataset(Dataset):
    """
    train_dict - dict of dicts: rle for 4 class for every image
    
    valid: bool, return or not file names (need for error visualization)
    """
    def __init__(self, files_path, train_dict, transform=None, exclude_images=[], valid=False):
        
        self.files_path = files_path
        self.train_dict = train_dict
        # exclude images
        img_list = os.listdir(files_path)
        for img in exclude_images:
            if img in img_list:
                img_list.remove(img)
        self.img_list = sorted(img_list)
        
        self.transform = transform
        self.valid = valid

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # fix seed
        _fix_seeds(SEED)
        
        # upload image (numpy array)
        image_id = self.img_list[idx]
        image = jpeg.JPEG(self.files_path + image_id).decode()
        
        # upload mask (rle string)
        masks_dict = self.train_dict[image_id]
        mask = []
        
        for cls in masks_dict:
            cur_rle = masks_dict[cls]
            if cur_rle != '':
                # get numpy array from string
                cur_mask = rle2mask(cur_rle)
                mask.append(cur_mask)
            else:
                cur_mask = np.zeros((256,1600), dtype=np.uint8)
                mask.append(cur_mask)
        mask = np.array(mask).swapaxes(0, 1).swapaxes(1,2)       
        
        if self.transform is not None:
            aug_item = self.transform(image=image, mask=mask)
        
        if not self.valid:
            return torch.from_numpy(aug_item['image']).permute(2, 0, 1), \
               torch.from_numpy(aug_item['mask']).permute(2, 0, 1).float()
        else:
            return image_id, \
                torch.from_numpy(aug_item['image']).permute(2, 0, 1), \
                torch.from_numpy(aug_item['mask']).permute(2, 0, 1).float()


class SteelTestDataset(Dataset):
    """
    Dataset for test prediction
    """
    def __init__(self, files_path, transform=None):
        
        self.files_path = files_path
        self.img_list = sorted(os.listdir(files_path))
        self.transform = transform
            
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        _fix_seeds(SEED)
        image_id = self.img_list[idx]
        image = jpeg.JPEG(self.files_path + image_id).decode()
        
        aug_item = {}
        
        if self.transform is not None:
            aug_item = self.transform(image=image)
        else:
            aug_item['image'] = image
            
        return image_id, torch.from_numpy(aug_item['image']).permute(2, 0, 1).float()

    
# Datasets for Classification

class SteelClassTrainDataset(Dataset):
    """
    Dataset calss for multilabe classification
    
    files_path: str: path to images
    train_dict: dict of dicts: rle for 4 class for every image
    transform: augmentations
    exclude_images: image ids to exclude from training (duplicates or ambiguous labeling)
    """
    def __init__(self, files_path, train_dict, transform=None, exclude_images=[]):
        
        self.files_path = files_path
        self.train_dict = train_dict
        # exclude images
        img_list = os.listdir(files_path)
        for img in exclude_images:
            if img in img_list:
                img_list.remove(img)
        self.img_list = sorted(img_list)
        
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # fix seed
        _fix_seeds(SEED)
        
        # upload image (numpy array)
        image_id = self.img_list[idx]
        image = jpeg.JPEG(self.files_path + image_id).decode()
        
        # upload mask (rle string)
        masks_dict = self.train_dict[image_id]
        labels = []
        
        for cls in masks_dict:
            cur_rle = masks_dict[cls]
            if cur_rle != '':
                # get numpy array from string
                labels.append(1)
            else:
                labels.append(0)
        labels = np.array(labels)      
        
        if self.transform is not None:
            aug_item = self.transform(image=image)

        return torch.from_numpy(aug_item['image']).permute(2, 0, 1), \
               torch.from_numpy(labels).float()
