# Trainer Class for Multilabel classificaiton
import sys

import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

from code.utils import _fix_seeds
from code.samplers import ClassProbSampler


def _init_fn(worker_id):
    _fix_seeds()
    
def batch_metrics_update(batch_mask, batch_preds, pred_thresholds):
    """
    batch_mask, batch_preds: torch.tensors of shape [BATCH_SIZE; NUM_CLASSES], NUM_CLASSES=4
    """
    batch_preds = torch.sigmoid(batch_preds)
    
    batch_preds = (batch_preds > pred_thresholds).float()
    batch_mask = (batch_mask > 0.5).float()
    
    tp_cls = (batch_mask * batch_preds).sum(0)
    #tn_cls = ((1-batch_mask) * (1 - batch_preds)).sum(0)
    fp_cls = ((1-batch_mask) * batch_preds).sum(0)
    fn_cls = (batch_mask * (1 - batch_preds)).sum(0)
    bs = batch_mask.shape[0] 
    
    acc_cls = ((batch_mask == batch_preds).sum(0).float()/bs).cpu().numpy()
    acc_batch = acc_cls.mean().item()
    
    precis_cls = (tp_cls/(tp_cls + fp_cls)).cpu().numpy()
    recall_cls = (tp_cls/(tp_cls + fn_cls)).cpu().numpy()
    
    return acc_batch, acc_cls, precis_cls, recall_cls#,\ 
            #tp_cls, tn_cls, fp_cls, fn_cls


class TrainerClassifier(object):
    """
    Class for train and validation
    """
    def __init__(self, 
                 path_to_save_model, 
                 loss, 
                 #metrics, # TODO: every metric as class
                 #optimizer, scheduler,
                 train_dataset, train_loader_params,
                 valid_dataset, valid_loader_params,
                 valid_fraction,
                 train_image_class,
                 device,
                 pred_thresholds = torch.tensor([0.3, 0.2, 0.5, 0.3]),
                 class_weights_sampler={0: 64/29., 1: 64/4., 2: 64/2., 3: 64/25., 4: 64/4.},
                 num_classes=4,
                 num_epochs=20):
        
        #torch.set_default_tensor_type("torch.FloatTensor")
        _fix_seeds()
        self.path_to_save_model = path_to_save_model
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.best_loss = float("inf")
        self.device = device
        self.model = None
        self.valid_fraction = valid_fraction
        
        self.loss = loss
        #self.metrics = metrics # TODO: every metric as class
        self.pred_thresholds = pred_thresholds.cuda()
        
        self.class_weights_sampler = class_weights_sampler
        self.train_image_class = train_image_class
        
        self.optimizer = None#optimizer
        self.init_lr = None#[pg['lr'] for pg in self.optimizer.param_groups][0]
        self.scheduler = None#scheduler
        # data
        self.datasets = {
            'train': train_dataset, # same datasets with different transforms
            'valid': valid_dataset
        }
        self.loader_params = {
            'train': train_loader_params,
            'valid': valid_loader_params
        }
        self.loaders = {
            'train': None, # create during cross validation
            'valid': None
        }
        self.samplers = {
            'train': None,# create during cross validation
            'valid': None
        }
        # metrics history
        self.loss_history = {
            'train': [],
            'valid': []
        }
        self.metric_history = {
            'train': [],
            'valid': []
        }
        self.epoch_additional_metrics = {
            'train': [],
            'valid': []            
        }
    
    def make_model(self, model_name, 
                   optimizer_params,#={'lr':1e-3, 'momentum':0.9, 'weight_decay':1e-6}, 
                   scheduler_params,
                   pretrained=True):
        """
        Load pretrained model for classification
        """
        model = torchvision.models.__dict__[model_name](pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, self.num_classes)
        
        #optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-6)
        #lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5, last_epoch=-1)
        optimizer = optim.SGD(model.parameters(), **optimizer_params)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
        
        return model, optimizer, lr_scheduler
    
    def make_loader(self, phase, sampler):
        """
        Make loader for current phase and sampler
        """
        params = self.loader_params[phase]
        dataset = self.datasets[phase]
        loader = DataLoader(dataset,
                            sampler=sampler, 
                            **params)
        return loader
        
    def forward_train(self, images, targets):
        """
        Forward pass for train phase
        """
        preds = self.model.forward(images)
        loss_ = self.loss(preds, targets)
        
        self.optimizer.zero_grad()
        loss_.backward()
        self.optimizer.step()
        
        return loss_, preds

    def forward_valid(self, images, targets):
        """
        Forward pass for validation phase
        """
        with torch.no_grad():
            preds = self.model.forward(images)
            loss_ = self.loss(preds, targets)
            
        return loss_, preds
    
    def _format_logs(self, logs):
        """
        logs - dict with metrics values
        """
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s 

    def run_epoch(self, phase):
        """
        Run single epoch of train or validation
        
        phase: str, 'train' or 'valid'
        """
        # enter mode
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
            
        # fix seeds
        _fix_seeds()   
        
        loader = self.loaders[phase]
        data_len = len(loader)
        
        logs = {}
        epoch_loss = 0 
        acc_epoch = 0
        acc_cls_arr = np.array([]).reshape(0,self.num_classes)
        precis_cls_arr = np.array([]).reshape(0,self.num_classes)
        recall_cls_arr = np.array([]).reshape(0,self.num_classes)
        #epoch_meter = Meter(phase, self.pred_threshold, self.device)
        
        # iterate over data
        with tqdm(loader, desc=phase, file=sys.stdout) as iterator:
            for (batch_img, batch_mask) in iterator:
                
                _fix_seeds()
                # to gpu
                batch_img = batch_img.to(self.device)
                batch_mask = batch_mask.to(self.device)
        
                if phase == 'train':
                    batch_loss, batch_preds = self.forward_train(batch_img, batch_mask)
                else:
                    batch_loss, batch_preds = self.forward_valid(batch_img, batch_mask)   
                
                #dice_batch = epoch_meter.update_batch(batch_mask, batch_preds)
                epoch_loss += batch_loss.item()
                batch_metrics = batch_metrics_update(batch_mask, batch_preds, self.pred_thresholds)

                logs['acc_batch'] = batch_metrics[0]
                acc_epoch += batch_metrics[0]
                acc_cls_arr = np.vstack((acc_cls_arr, batch_metrics[1]))
                precis_cls_arr = np.vstack((precis_cls_arr, batch_metrics[2]))
                recall_cls_arr = np.vstack((recall_cls_arr, batch_metrics[3]))
                
                del batch_mask, batch_img, batch_preds, batch_metrics        
                #torch.cuda.empty_cache()
                
                # save current batch loss value for output
                loss_logs = {self.loss.__name__: batch_loss.item()}
                logs.update(loss_logs)
                s = self._format_logs(logs)
                iterator.set_postfix_str(s)
        
        #dice_mean, epoch_dices = epoch_meter.update_epoch()
        #epoch_cnt_neg_pred, epoch_cnt_neg_mask = epoch_meter.update_epoch()
        #epoch_cnt_pos_pred, epoch_cnt_pos_mask, \
        #epoch_precision_batch, epoch_recall_batch = epoch_meter.update_epoch()
        acc_epoch /= data_len
        acc_cls_arr = acc_cls_arr.mean(0)
        precis_cls_arr = precis_cls_arr.mean(0)
        recall_cls_arr = recall_cls_arr.mean(0)
        
        epoch_loss /= data_len
        self.scheduler.step()
        #self.scheduler.step(epoch_loss)
        
        #del epoch_meter
        torch.cuda.empty_cache()
        
        return epoch_loss, acc_epoch, acc_cls_arr,\
               precis_cls_arr, recall_cls_arr
        
    def run_model(self, model, optimizer, scheduler, fold_num=None):
        """
        Iterate through epochs with both train and validation pahses
        """
        _fix_seeds()
        cur_time = datetime.now().strftime("%H:%M:%S")
        # initialize model, learning rate and loss for each run
        self.best_loss = float("inf")
        self.model = model
        self.model.to(self.device)
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.init_lr = [pg['lr'] for pg in self.optimizer.param_groups][0]
        
        for g in self.optimizer.param_groups:
            g['lr'] = self.init_lr
        
        if fold_num:
            temp_str = self.path_to_save_model
            temp_lst = temp_str.split('.')
            fold_num_str = '_' + str(fold_num) + '.'
            model_path = ''.join(['..'] + temp_lst[:-1] + [fold_num_str] + [temp_lst[-1]])
        else:
            model_path = self.path_to_save_model
            
            # generate train and valid indices
            data_size = len(self.datasets['train'])
            val_split = int(np.floor((self.valid_fraction) * data_size))
            indices = list(range(data_size))
            np.random.shuffle(indices)
            valid_indices, train_indices = indices[:val_split], indices[val_split:]
            inds_dict = {'train':train_indices, 'valid':valid_indices}
            
            # create samplers and loaders
            for phase in ['train', 'valid']:
                self.samplers[phase] = ClassProbSampler(inds_dict[phase], 
                                                        self.class_weights_sampler, 
                                                        self.train_image_class)
                self.loaders[phase] = self.make_loader(phase, self.samplers[phase])
        
        for epoch in range(self.num_epochs):
            _fix_seeds()
            
            print(f"Starting epoch: {epoch} | time: {cur_time}")
            print('LR:',[pg['lr'] for pg in self.optimizer.param_groups])
            for phase in ['train', 'valid']:
                
                epoch_all_metrics = self.run_epoch(phase)
                self.loss_history[phase].append(epoch_all_metrics[0])
                self.metric_history[phase].append(epoch_all_metrics[1])
                self.epoch_additional_metrics[phase].append(epoch_all_metrics[2:])
                
                if phase == 'valid':
                    print(f'Valid avg loss: {epoch_all_metrics[0]}')
                    print(f'Valid avg accuracy: {epoch_all_metrics[1]}')
                
                del epoch_all_metrics
                torch.cuda.empty_cache()
                
            if self.best_loss > self.loss_history['valid'][-1]:
                self.best_loss = self.loss_history['valid'][-1]
                torch.save(self.model, model_path)
                print('*** Model saved! ***\n')
                
            cur_time = datetime.now().strftime("%H:%M:%S")
            
            
    def cross_val_score(self,
                        model_name,
                        optimizer_params,
                        scheduler_params,
                        cv_schema):
        """
        Make N folds cross-validation and average predictions from every fold
        """
        _fix_seeds()
        
        dataset_len = len(self.datasets['train']) # does not matter train ot valid if they are equal, except transforms
        indices = list(range(dataset_len))
        
        for fold_num, (train_indices, valid_indices) in enumerate(cv_schema.split(indices)):
            
            print('*******************************************')
            print('Starting fold {}'.format(fold_num))
        
            # make samplers and loaders        
            inds_dict = {'train':train_indices, 'valid':valid_indices}
            for phase in ['train', 'valid']:
                self.samplers[phase] = ClassProbSampler(inds_dict[phase], 
                                                        self.class_weights_sampler, 
                                                        self.train_image_class)
                self.loaders[phase] = self.make_loader(phase, self.samplers[phase])

            # initialize model and process train and validation for this fold
            model, optimizer, lr_scheduler = self.make_model(model_name, 
                                                             optimizer_params,
                                                             scheduler_params,
                                                             pretrained=True)
            self.run_model(model, optimizer, lr_scheduler, fold_num=str(fold_num))
            
            print('Fold {} finished!\n'.format(fold_num))
            
            
    @staticmethod
    def evaluate_model(best_model,
                       valid_dataset,
                       valid_sampler,
                       device,
                       cls_thresholds={1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5},
                       cls_min_area={1: 500, 2: 500, 3: 2000, 4: 2000},
                       batch_size=16,
                       num_workers=8):
        """
        Model validation with post processing
        """        
        _fix_seeds()
        # load saved model
        best_model.to(device)
        best_model.eval()
        
        # create validation loader
        loader = DataLoader(valid_dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          sampler=valid_sampler,
                          worker_init_fn=_init_fn,
                          pin_memory=False,
                          num_workers=num_workers)
        
        dices = np.array([])
        
        # iterate over data
        with tqdm(loader, desc='valid', file=sys.stdout) as iterator:
            for (batch_img, batch_mask) in iterator:
                # to gpu
                batch_img = batch_img.to(device)
                #batch_mask = batch_mask.to(device)
                        
                batch_preds = best_model(batch_img)
                batch_preds = torch.sigmoid(batch_preds)   
                batch_preds = batch_preds.detach().cpu().numpy()
                batch_mask = batch_mask.detach().numpy()
                
                batch_preds_processed = []
                for preds in batch_preds:
                    for cls, pred in enumerate(preds):
                        pred = post_process(pred, cls_thresholds[cls + 1], cls_min_area[cls + 1])
                        batch_preds_processed.append(pred)
                
                batch_preds_processed = np.array(batch_preds_processed)
                
                bs = batch_preds.shape[0]
                w = batch_preds.shape[2]
                h = batch_preds.shape[3]

                # 4 classes - 4 rows for every image
                batch_preds_processed = batch_preds_processed.reshape((bs * 4, w, h))
                batch_mask = batch_mask.reshape((bs * 4, w, h))
                
                dices_batch = dice_coeff_batch(batch_preds_processed, batch_mask)
                dices = np.append(dices, dices_batch)
        
        # avg dice on all validation
        return np.mean(dices), dices
