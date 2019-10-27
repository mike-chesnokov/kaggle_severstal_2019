import numpy as np
import torch

from code.metrics import dice_coeff_mean


class Meter(object):
    """
    Class for metrics computing and monitoring
    """
    def __init__(self, phase, pred_threshold, device):
        self.phase = phase
        self.pred_threshold = pred_threshold
        self.device = device
        
        self.epoch_dices = []
        #self.epoch_cnt_neg_pred = []
        #self.epoch_cnt_neg_mask = []
        #self.epoch_cnt_pos_pred = []
        #self.epoch_cnt_pos_mask = []
        #self.epoch_precision_batch = []
        #self.epoch_recall_batch = []        
        
        #if self.phase == 'valid':
        #    self.predictions = torch.tensor(np.array([]), device=self.device, dtype=torch.float32)
        #    self.ground_truth = torch.tensor(np.array([]), device=self.device, dtype=torch.float32)
        #else:
        #    self.predictions = None
        #    self.ground_truth = None
        
    
    def update_batch(self, batch_mask, batch_preds):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        batch_preds = torch.sigmoid(batch_preds)
        
        #compute additional metrics for monitoring
        #cnt_neg_pred, cnt_neg_mask, \
        #cnt_pos_pred, cnt_pos_mask, \
        #precision_batch, recall_batch = self.compute_additional_metrics(batch_preds, batch_mask)
        #cnt_neg_pred, cnt_neg_mask = self.compute_additional_metrics(batch_preds, batch_mask) 
        
        # flat predictions and masks
        bs = batch_preds.shape[0]
        w = batch_preds.shape[2]
        h = batch_preds.shape[3]
                
        # 4 classes - 4 rows for every image
        preds_flat = batch_preds.view((bs * 4, w, h))
        mask_flat = batch_mask.view((bs * 4, w, h))
        # binarization
        preds_flat = (preds_flat > self.pred_threshold).float()
        mask_flat = (mask_flat > 0.5).float()
                
        #if self.phase == 'valid':
        #    self.predictions = torch.cat((self.predictions, preds_flat))
        #    self.ground_truth = torch.cat((self.ground_truth, mask_flat))
                
        # avg dice on batch
        dice_batch = dice_coeff_mean(preds_flat, mask_flat)
        
        self.epoch_dices.append(dice_batch)
        #self.epoch_cnt_neg_pred.append(cnt_neg_pred)
        #self.epoch_cnt_neg_mask.append(cnt_neg_mask)
        #self.epoch_cnt_pos_pred.append(cnt_pos_pred)
        #self.epoch_cnt_pos_mask.append(cnt_pos_mask)
        #self.epoch_precision_batch.append(precision_batch)
        #self.epoch_recall_batch.append(recall_batch)         
        
        del batch_mask, batch_preds, preds_flat, mask_flat  
        
        return dice_batch
    
    
    def update_epoch(self):
        
        #if self.phase == 'valid':
        #    dice_mean = dice_coeff_mean(self.predictions, self.ground_truth)
        #    self.predictions = None
        #    self.ground_truth = None
        #else:
        dice_mean = np.mean(self.epoch_dices)
            
        return dice_mean, self.epoch_dices#, \
               #self.epoch_cnt_neg_pred, self.epoch_cnt_neg_mask#, \
               #self.epoch_cnt_pos_pred, self.epoch_cnt_pos_mask, \
               #self.epoch_precision_batch, self.epoch_recall_batch
    
    def compute_additional_metrics(self, batch_preds, batch_mask):
        """
        Computes metrics for batch monitoring.

        Params:
            batch_preds: torch.Tensor: predictions for batch of shape (BS, num_classes, Width, Height)
            batch_mask: torch.Tensor: masks for batch
        
        Returns:
            cnt_neg_pred: int: number of neg channels (all zeros) of preds in batch
            cnt_neg_mask: int: number of neg channels (all zeros) of masks in batch
            cnt_pos_pred: list: number of pos channels (at least 1 non zero) of preds in batch for every class
            cnt_pos_mask: list: number of pos channels (at least 1 non zero) of preds in batch for every class
            precision_batch: list: avg precision over batch for every class
            recall_batch: list: avg recall over batch for every class
        """
        batch_preds = (batch_preds > self.pred_threshold).float()
        batch_mask = (batch_mask > 0.5).float()
        
        m1 = batch_preds.view(64, 4, -1).sum(2)
        m2 = batch_mask.view(64, 4, -1).sum(2)

        # number of neg masks in batch
        cnt_neg_pred = (m1.sum(1) == 0).sum().item()
        cnt_neg_mask = (m2.sum(1) == 0).sum().item()
        '''
        # number of pos masks of each class
        cnt_pos_pred = [(m1 > 0).sum(0)[x].item() for x in range(4)]
        cnt_pos_mask = [(m2 > 0).sum(0)[x].item() for x in range(4)]

        # for every class calculate TP, TN, FP, FN pixels
        tp = (batch_preds == 1.) * (batch_mask == 1.)
        tn = (batch_preds == 0.) * (batch_mask == 0.)
        fp = (batch_preds == 1.) * (batch_mask == 0.)
        fn = (batch_preds == 0.) * (batch_mask == 1.)

        tp_batch = tp.view(64, 4, -1).sum(2).float()
        tn_batch = tn.view(64, 4, -1).sum(2).float()
        fp_batch = fp.view(64, 4, -1).sum(2).float()
        fn_batch = fn.view(64, 4, -1).sum(2).float()

        # calculate avg precision and recall over batch
        precision_batch = (tp_batch/(tp_batch + fp_batch)).mean(0)
        recall_batch = (tp_batch/(tp_batch + fn_batch)).mean(0)
        
        precision_batch = [precision_batch[x].item() for x in range(4)]
        recall_batch = [recall_batch[x].item() for x in range(4)]
        '''
        return  cnt_neg_pred, cnt_neg_mask#, \
                #cnt_pos_pred, cnt_pos_mask, \
                #precision_batch, recall_batch
