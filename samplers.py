import os

import torch
from torch.utils.data import Sampler

from configs import (
    TRAIN_PATH,
)

class SubsetSequentSampler(Sampler):
    """Samples elements with given indices sequentially

    Arguments:
        data_source (Dataset): dataset to sample from
        indices (ndarray): indices of the samples to take
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices) 
    
    
class SubsetRandomSampler(Sampler):
    """Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    
class ClassProbSampler(Sampler):
    """
    Samples elements randomly from a given list of indices 
    for imbalanced dataset with replacement
    
    Params:
        indices: list: a list of indices
        class_weights: dict: weight for each class
        files_path: path to files
        train_image_class: dict: image name -> class
        num_samples: int, optional: number of samples to draw
    """

    def __init__(self, indices, 
                 class_weights,
                 train_image_class,
                 with_replacement=False,
                 files_path=TRAIN_PATH,
                 num_samples=None):
                           
        self.indices = indices
        self.with_replacement = with_replacement
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
        
        img_list = sorted(os.listdir(files_path))
        self.weights = torch.DoubleTensor(
            [class_weights[train_image_class[img_list[ind]]] \
             for ind in indices])
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
                self.weights, self.num_samples, replacement=self.with_replacement))

    def __len__(self):
        return self.num_samples
    