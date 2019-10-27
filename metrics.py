# Classes for metrics
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def dice_coeff_mean(pred, target):
    """
    Copmputes mean dice coef over image_class rows.
    Equals 1 where both mask and preds all zeros
    """
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten   

    # calculate dice
    #intersection = (m1 * m2).sum(1)
    dice = (2. * (m1 * m2).sum(1)) / (m1.sum(1) + m2.sum(1))
    
    # get index where mask and predictions all zeros
    dice_one_inds = ((m1.sum(1)==0) * (m2.sum(1)==0)).nonzero()    
    # make dice = 1 where preds and mask all zeros
    if dice_one_inds.shape[0] > 0:
        dice[dice_one_inds] = 1
    
    return dice.float().mean().item()


class SoftDiceMetric(nn.Module):

    __name__ = 'dice_soft'
    
    def __init__(self, smooth=1.):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = targets.size(0)  # Number of batches

        score = self.dice_coeff(probs, targets, self.smooth)
        score = score.sum() / num
        #score = 1 - score.sum() / num
        return score
    
    def dice_coeff(self, pred, target, smooth=1.):

        num = pred.size(0)
        m1 = pred.view(num, -1)  # Flatten
        m2 = target.view(num, -1)  # Flatten
        intersection = (m1 * m2).sum()

        return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

    
class HardDiceMetric(nn.Module):

    __name__ = 'dice_hard'
    
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = targets.size(0)  # Number of batches

        score = self.dice_coeff(probs, targets)
        score = score.sum() / num
        #score = 1 - score.sum() / num
        return score
    
    def dice_coeff(self, pred, target):

        num = pred.size(0)
        m1 = pred.view(num, -1)  # Flatten
        m2 = target.view(num, -1)  # Flatten
        
        if m1.sum().item() == 0 and m2.sum().item() == 0:
            return 1.

        intersection = (m1 * m2).sum()

        return (2. * intersection) / (m1.sum() + m2.sum())