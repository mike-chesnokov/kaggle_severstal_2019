# Pytorch Losses for segmentation
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEWeightedLoss(nn.Module):
    """
    Loss for segmentation
    """
    __name__ = 'bce_weighted_loss'
    
    def __init__(self, class_weights={1: 1, 2: 1, 3: 1, 4: 1}):
        super().__init__()
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        bs = inputs.shape[0]
        # w = inputs.shape[2]
        # h = inputs.shape[3]
        
        loss_all = 0
        
        for cls in self.class_weights:
            cur_inputs = inputs[:, cls-1, :, :].view(bs, -1)
            cur_targets = targets[:, cls-1, :, :].view(bs, -1)

            BCE = F.binary_cross_entropy(cur_inputs, cur_targets, reduction='mean')            
            loss_all += (BCE * self.class_weights[cls])
        
        return loss_all

    
class BCEWeightedClassifLoss(nn.Module):
    """
    Loss for classification
    """
    __name__ = 'bce_weighted_loss'
    
    def __init__(self, class_weights={1: 1, 2: 1, 3: 1, 4: 1}):
        super().__init__()
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        bs = inputs.shape[0]
        loss_all = 0
        
        for cls in self.class_weights:
            cur_inputs = inputs[:, cls-1]
            cur_targets = targets[:, cls-1]

            BCE = F.binary_cross_entropy(cur_inputs, cur_targets, reduction='mean')            
            loss_all += (BCE * self.class_weights[cls])
        
        return loss_all
    
    
class BCEDiceWeightedLoss(nn.Module):
    __name__ = 'bce_dice_weighted_loss'
    
    def __init__(self, smooth=1, dice_loss_weight=1, class_weights={1: 1, 2: 1, 3: 1, 4:1 }):
        super().__init__()
        self.smooth = smooth
        self.class_weights = class_weights
        self.dice_loss_weight = dice_loss_weight

    def forward(self, inputs, targets):
        
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        bs = inputs.shape[0]
        loss_all = 0
        
        for cls in self.class_weights:
            cur_inputs = inputs[:, cls-1, :, :].view(bs, -1)
            cur_targets = targets[:, cls-1, :, :].view(bs, -1)
        
            intersection = (cur_inputs * cur_targets).sum()                            
            dice_loss = 1 - (2.*intersection + self.smooth)/(cur_inputs.sum() + cur_targets.sum() + self.smooth)

            BCE = F.binary_cross_entropy(cur_inputs, cur_targets, reduction='mean')
            cur_Dice_BCE = BCE + dice_loss * self.dice_loss_weight
            
            loss_all += (cur_Dice_BCE * self.class_weights[cls])
        
        return loss_all


class TverskyWeightedLoss(nn.Module):
    """
    With alpha=beta=0.5, this loss becomes equivalent to Dice Loss
    """
    __name__ = 'tversky_weighted_loss'
    
    def __init__(self, alpha=0.5, beta=0.5, smooth=1, class_weights={1: 1, 2: 1, 3: 1, 4: 1}):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)    
        
        bs = inputs.shape[0]
        loss_all = 0
        
        for cls in self.class_weights:
            # flatten label and prediction tensors
            cur_inputs = inputs[:, cls-1, :, :].view(bs, -1)
            cur_targets = targets[:, cls-1, :, :].view(bs, -1)
                    
            # True Positives, False Positives & False Negatives
            TP = (cur_inputs * cur_targets).sum()    
            FP = ((1-cur_targets) * cur_inputs).sum()
            FN = (cur_targets * (1 - cur_inputs)).sum()

            Tversky = 1 - (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
            
            loss_all += (Tversky * self.class_weights[cls])
        
        return loss_all
    
    
class DiceLoss(nn.Module):
    __name__ = 'dice_loss'
    
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)
        
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + self.smooth)/(inputs.sum() + targets.sum() + self.smooth)  
        
        return 1 - dice


class BCELogitsLoss(nn.Module):
    __name__ = 'bce_logits_loss'
    
    def __init__(self,):
        super().__init__()

    def forward(self, inputs, targets):
        
        loss = nn.BCEWithLogitsLoss()
        
        return loss(inputs, targets)    
    

class BCEDiceLoss(nn.Module):
    __name__ = 'bce_dice_loss'
    
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + self.smooth)/(inputs.sum() + targets.sum() + self.smooth)
        
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
    

class IoULoss(nn.Module):
    __name__ = 'iou_loss'
    
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + self.smooth)/(union + self.smooth)
                
        return 1 - IoU
    
    
class FocalLoss(nn.Module):
    """
    Loss for extremely imbalanced datasets where positive cases were relatively rare.
    """
    __name__ = 'focal_loss'
    
    def __init__(self, alpha=0.8, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        
        focal_loss = self.alpha * (1-BCE_EXP)**self.gamma * BCE
                       
        return focal_loss
    

class TverskyLoss(nn.Module):
    """
    With alpha=beta=0.5, this loss becomes equivalent to Dice Loss
    """
    __name__ = 'tversky_loss'
    
    def __init__(self, alpha=0.5, beta=0.5, smooth=1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)  
        
        return 1 - Tversky


class FocalTverskyLoss(nn.Module):
    """
    Variation of Tversky loss with the gamma modifier from Focal Loss
    """
    __name__ = 'focal_tversky_loss'
    
    def __init__(self, alpha=0.5, beta=0.5, smooth=1, gamma=1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.gamma = gamma

    def forward(self, inputs, targets):
        
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)  
        FocalTversky = (1 - Tversky)**self.gamma
                       
        return FocalTversky
