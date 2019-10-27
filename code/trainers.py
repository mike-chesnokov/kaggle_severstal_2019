import sys

import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from datetime import datetime
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader

from code.utils import _fix_seeds, mask2rle
from code.metrics import dice_coeff_mean
from code.meter import Meter
from code.samplers import ClassProbSampler, SubsetRandomSampler, SubsetSequentSampler


def _init_fn(worker_id):
    _fix_seeds()
    

def dice_coeff_batch(pred, target):
    """
    Copmputes dice coefs for batch over image_class rows.
    Equals 1 where both mask and preds all zeros
    
    pred: np.array
    target: np.array
    """
    num = pred.shape[0]
    m1 = pred.reshape(num, -1)  # Flatten
    m2 = target.reshape(num, -1)  # Flatten   

    # calculate dice
    # intersection = (m1 * m2).sum(1)
    dice = (2. * (m1 * m2).sum(1)) / (m1.sum(1) + m2.sum(1))
    
    # get index where mask and predictions all zeros
    dice_one_inds = np.transpose(((m1.sum(1) == 0) * (m2.sum(1) == 0)).nonzero())
    # make dice = 1 where preds and mask all zeros
    if dice_one_inds.shape[0] > 0:
        dice[dice_one_inds] = 1
    
    return dice


def post_process(probability, threshold, min_size):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
        
    probability: np.array
    threshold: int
    min_size: int
    """
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.float32)

    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1

    return predictions


class Trainer(object):
    """
    Class for train and validation
    """
    def __init__(self, 
                 path_to_save_model, 
                 loss, 
                 #metrics, # TODO: every metric as class
                 train_dataset, train_loader_params, train_sampler_params,
                 valid_dataset, valid_loader_params, valid_sampler_params,
                 valid_fraction,
                 #train_image_class,
                 device,
                 pred_threshold,
                 #class_weights_sampler={0: 64/29., 1: 64/4., 2: 64/2., 3: 64/25., 4: 64/4.},
                 num_classes=4,
                 num_epochs=20):
        
        #torch.set_default_tensor_type("torch.FloatTensor")
        _fix_seeds()
        self.path_to_save_model = path_to_save_model
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.best_loss = float("inf")
        self.device = device
        self.valid_fraction = valid_fraction
        
        self.loss = loss
        #self.metrics = metrics # TODO: every metric as class
        self.pred_threshold = pred_threshold
        
        self.model = None
        self.optimizer = None  #optimizer
        self.init_lr = None  #[pg['lr'] for pg in self.optimizer.param_groups][0]
        self.scheduler = None   #scheduler
        # data
        self.datasets = {
            'train': train_dataset, # same datasets with different transforms
            'valid': valid_dataset
        }
        self.loader_params = {
            'train': train_loader_params,
            'valid': valid_loader_params
        }
        self.sampler_params = {
            'train': train_sampler_params,
            'valid': valid_sampler_params,
        }
        self.loaders = {
            'train': None,  # create during cross validation
            'valid': None
        }
        self.samplers = {
            'train': None,  # create during cross validation
            'valid': None
        }
        # metrics history
        self.loss_history = {
            'train': [],
            'valid': []
        }
        self.dice_history = {
            'train': [],
            'valid': []
        }
        self.epoch_additional_metrics = {
            'train': [],
            'valid': []            
        }

    def make_model(self, model_params, 
                   optimizer_params, 
                   scheduler_params):
        """
        Load pretrained model for classification
        
        model_params: dict, 'model_name' in ['unet', 'fpn']; 'encoder_name' in pretrained_models; 
                    activation, encoder_weights in ['imagenet'];
        optimizer_params: dict, 'optimizer_name' in ['sgd', 'adam']
        scheduler_params: dict, 'scheduler_name' in ['cos', 'step', 'redonplat']
        """
        model_params_copy = model_params.copy()
        optimizer_params_copy = optimizer_params.copy()
        scheduler_params_copy = scheduler_params.copy()
        
        model_name = model_params_copy.pop('model_name', None)
        optimizer_name = optimizer_params_copy.pop('optimizer_name', None)
        scheduler_name = scheduler_params_copy.pop('scheduler_name', None)
        
        # set model
        if model_name == 'unet':
            model = smp.Unet(
                classes=self.num_classes,
                **model_params_copy
            )
            
        elif model_name == 'fpn':
            model = smp.FPN(
                classes=self.num_classes,
                **model_params_copy
            )  
        else:
            raise ValueError('***Error: model type not implemented!***')
        
        # set optimizer
        if optimizer_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), **optimizer_params_copy)
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), **optimizer_params_copy)
        else:
            raise ValueError('***Error: optimizer type not implemented!***')
            
        # set scheduler
        if scheduler_name == 'cos':
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params_copy)
        elif scheduler_name == 'step':
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_params_copy)
        elif scheduler_name == 'redonplat':
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params_copy)
        else:
            raise ValueError('***Error: scheduler type not implemented!***') 
        
        return model, optimizer, lr_scheduler        
        
    def make_sampler(self, phase, indices):
        """
        Make sampler from current indices
        """
        params = self.sampler_params[phase]
        params_copy = params.copy()
        sampler_name = params_copy.pop('sampler_name', None)
        
        if sampler_name == 'ClassProbSampler':
            sampler = ClassProbSampler(indices, **params_copy)
        elif sampler_name == 'SubsetRandomSampler':
            sampler = SubsetRandomSampler(indices, **params_copy)   
        elif sampler_name == 'SubsetSequentSampler':
            sampler = SubsetSequentSampler(indices, **params_copy)
        else:
            raise ValueError('***Error: sampler type not implemented!***') 
            
        return sampler
    
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
        epoch_meter = Meter(phase, self.pred_threshold, self.device)
        
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
                
                dice_batch = epoch_meter.update_batch(batch_mask, batch_preds)
                epoch_loss += batch_loss.item()

                logs['dice_batch'] = dice_batch
                
                del batch_mask, batch_img, batch_preds#, preds_flat, mask_flat         
                #torch.cuda.empty_cache()
                
                # save current batch loss value for output
                loss_logs = {self.loss.__name__: batch_loss.item()}
                logs.update(loss_logs)
                s = self._format_logs(logs)
                iterator.set_postfix_str(s)
        
        dice_mean, epoch_dices = epoch_meter.update_epoch()
        #epoch_cnt_neg_pred, epoch_cnt_neg_mask = epoch_meter.update_epoch()
        #epoch_cnt_pos_pred, epoch_cnt_pos_mask, \
        #epoch_precision_batch, epoch_recall_batch = epoch_meter.update_epoch()
        
        epoch_loss /= data_len
        self.scheduler.step()
        #self.scheduler.step(epoch_loss)
        
        del epoch_meter
        torch.cuda.empty_cache()
        
        return epoch_loss, dice_mean, epoch_dices#, \
               #epoch_cnt_neg_pred, epoch_cnt_neg_mask#, \
               #epoch_cnt_pos_pred, epoch_cnt_pos_mask, \
               #epoch_precision_batch, epoch_recall_batch   
        
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
            val_split = int(np.floor(self.valid_fraction * data_size))
            indices = list(range(data_size))
            np.random.shuffle(indices)
            valid_indices, train_indices = indices[:val_split], indices[val_split:]
            inds_dict = {'train': train_indices, 'valid': valid_indices}
            
            # create samplers and loaders
            for phase in ['train', 'valid']:
                self.samplers[phase] = self.make_sampler(phase, inds_dict[phase])
                #self.samplers[phase] = ClassProbSampler(inds_dict[phase], 
                #                                        self.class_weights_sampler, 
                #                                        self.train_image_class)
                self.loaders[phase] = self.make_loader(phase, self.samplers[phase])
        
        for epoch in range(self.num_epochs):
            _fix_seeds()
            
            print(f"Starting epoch: {epoch} | time: {cur_time}")
            print('LR:',[pg['lr'] for pg in self.optimizer.param_groups])
            for phase in ['train', 'valid']:
                
                epoch_all_metrics = self.run_epoch(phase)
                self.loss_history[phase].append(epoch_all_metrics[0])
                self.dice_history[phase].append(epoch_all_metrics[1])
                self.epoch_additional_metrics[phase].append(epoch_all_metrics[2:])
                
                if phase == 'valid':
                    print(f'Valid avg loss: {epoch_all_metrics[0]}')
                    print(f'Valid avg dice: {epoch_all_metrics[1]}')
                
                del epoch_all_metrics
                torch.cuda.empty_cache()
                
            if self.best_loss > self.loss_history['valid'][-1]:
                self.best_loss = self.loss_history['valid'][-1]
                torch.save(self.model, model_path)
                print('*** Model saved! ***\n')
                
            cur_time = datetime.now().strftime("%H:%M:%S")

    def cross_val_score(self,
                        model_params,
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
                self.samplers[phase] = self.make_sampler(phase, inds_dict[phase])
                #self.samplers[phase] = ClassProbSampler(inds_dict[phase], 
                #                                        self.class_weights_sampler, 
                #                                        self.train_image_class)
                self.loaders[phase] = self.make_loader(phase, self.samplers[phase])

            # initialize model and process train and validation for this fold
            model, optimizer, lr_scheduler = self.make_model(model_params, 
                                                             optimizer_params,
                                                             scheduler_params)
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
        preds_rle = []
        
        # iterate over data
        with tqdm(loader, desc='valid', file=sys.stdout) as iterator:
            for (fnames, batch_img, batch_mask) in iterator:
                # to gpu
                batch_img = batch_img.to(device)
                #batch_mask = batch_mask.to(device)
                        
                batch_preds = best_model(batch_img)
                batch_preds = torch.sigmoid(batch_preds)   
                batch_preds = batch_preds.detach().cpu().numpy()
                batch_mask = batch_mask.detach().numpy()
                        
                batch_preds_processed = []
                for fname, preds in zip(fnames, batch_preds):
                    for cls, pred in enumerate(preds):
                        pred = post_process(pred, cls_thresholds[cls + 1], cls_min_area[cls + 1])
                        batch_preds_processed.append(pred)
                        # collect rle for every class
                        rle = mask2rle(pred)
                        name = fname + f"_{cls+1}"
                        preds_rle.append([name, rle])
                        
                batch_preds_processed = np.array(batch_preds_processed)
                
                bs = batch_preds.shape[0]
                w = batch_preds.shape[2]
                h = batch_preds.shape[3]

                # 4 classes - 4 rows for every image
                batch_preds_processed = batch_preds_processed.reshape((bs * 4, w, h))
                batch_mask = batch_mask.reshape((bs * 4, w, h))
                
                dices_batch = dice_coeff_batch(batch_preds_processed, batch_mask)
                dices = np.append(dices, dices_batch)
                
                #for fname, preds in zip(fnames, batch_preds_processed):
                #    for cls, pred in enumerate(preds):
                #        rle = mask2rle(pred)
                #        name = fname + f"_{cls+1}"
                #        preds_rle.append([name, rle])
                #del dices_batch, batch_mask, batch_preds_processed, fnames
                
        # avg dice on all validation
        return np.mean(dices), dices, preds_rle
            
        
##########################################################################
# !!!OLD VERSIONS!!!

class Trainer_old(object):
    """
    Class for train and validation
    """
    def __init__(self, 
                 model, path_to_save_model, 
                 loss, 
                 #metrics, # TODO: every metric as class
                 optimizer, scheduler,
                 pred_threshold,
                 train_loader,
                 valid_loader, 
                 device,
                 num_classes=4,
                 num_epochs=20):
        
        #torch.set_default_tensor_type("torch.FloatTensor")
        self.path_to_save_model = path_to_save_model
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.best_loss = float("inf")
        self.device = device
        _fix_seeds()
        self.model = model
        self.model.to(self.device)
        
        self.loss = loss
        #self.metrics = metrics # TODO: every metric as class
        self.pred_threshold = pred_threshold
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloaders = {
            'train': train_loader,
            'valid': valid_loader
        }

        # metrics history
        self.loss_history = {
            'train': [],
            'valid': []
        }
        self.dice_history = {
            'train': [],
            'valid': []
        }
        self.epoch_additional_metrics = {
            'train': [],
            'valid': []            
        }
        
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
        
        loader = self.dataloaders[phase]
        data_len = len(loader)
        
        logs = {}
        epoch_loss = 0
        epoch_meter = Meter(phase, self.pred_threshold, self.device)
        
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
                
                dice_batch = epoch_meter.update_batch(batch_mask, batch_preds)
                epoch_loss += batch_loss.item()

                logs['dice_batch'] = dice_batch
                
                del batch_mask, batch_img, batch_preds#, preds_flat, mask_flat         
                #torch.cuda.empty_cache()
                
                # save current batch loss value for output
                loss_logs = {self.loss.__name__: batch_loss.item()}
                logs.update(loss_logs)
                s = self._format_logs(logs)
                iterator.set_postfix_str(s)
        
        dice_mean, epoch_dices = epoch_meter.update_epoch()
        #epoch_cnt_neg_pred, epoch_cnt_neg_mask = epoch_meter.update_epoch()
        #epoch_cnt_pos_pred, epoch_cnt_pos_mask, \
        #epoch_precision_batch, epoch_recall_batch = epoch_meter.update_epoch()
        
        epoch_loss /= data_len
        #self.scheduler.step(epoch_loss)
        self.scheduler.step()
        
        del epoch_meter
        torch.cuda.empty_cache()
        
        return epoch_loss, dice_mean, epoch_dices#, \
               #epoch_cnt_neg_pred, epoch_cnt_neg_mask#, \
               #epoch_cnt_pos_pred, epoch_cnt_pos_mask, \
               #epoch_precision_batch, epoch_recall_batch   
        
    def run_model(self):
        """
        Iterate through epochs with both train and validation pahses
        """
        cur_time = datetime.now().strftime("%H:%M:%S")
        
        _fix_seeds()
        
        for epoch in range(self.num_epochs):
            _fix_seeds()
            
            print(f"Starting epoch: {epoch} | time: {cur_time}")
            print('LR:',[pg['lr'] for pg in self.optimizer.param_groups])
            for phase in ['train', 'valid']:
                
                epoch_all_metrics = self.run_epoch(phase)
                self.loss_history[phase].append(epoch_all_metrics[0])
                self.dice_history[phase].append(epoch_all_metrics[1])
                self.epoch_additional_metrics[phase].append(epoch_all_metrics[2:])
                
                if phase == 'valid':
                    print(f'Valid avg loss: {epoch_all_metrics[0]}')
                    print(f'Valid avg dice: {epoch_all_metrics[1]}')
                
                del epoch_all_metrics
                torch.cuda.empty_cache()
                
            if self.best_loss > self.loss_history['valid'][-1]:
                self.best_loss = self.loss_history['valid'][-1]
                torch.save(self.model, self.path_to_save_model)
                print('*** Model saved! ***\n')
                
            cur_time = datetime.now().strftime("%H:%M:%S")

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


class Trainer_old2(object):
    """
    Class for train and validation
    """
    def __init__(self, 
                 model, path_to_save_model, 
                 loss, 
                 #metrics, # TODO: every metric as class
                 optimizer, scheduler,
                 pred_threshold,
                 train_loader,
                 valid_loader, 
                 device,
                 num_classes=4,
                 num_epochs=20):
        
        #torch.set_default_tensor_type("torch.FloatTensor")
        self.path_to_save_model = path_to_save_model
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.best_loss = float("inf")
        self.device = device
        _fix_seeds()
        self.model = model
        self.model.to(self.device)
        
        self.loss = loss
        #self.metrics = metrics # TODO: every metric as class
        self.pred_threshold = pred_threshold
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloaders = {
            'train': train_loader,
            'valid': valid_loader
        }

        # metrics history
        self.loss_history = {
            'train': [],
            'valid': []
        }
        self.dice_history = {
            'train': [],
            'valid': []
        }
        
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
        
        loader = self.dataloaders[phase]
        data_len = len(loader)
        
        logs = {}
        epoch_loss = 0
        epoch_dice = []
        # for mean dice calculation over all data
        if phase == 'valid':
            predictions = torch.tensor(np.array([]), device=self.device, dtype=torch.float32)
            ground_truth = torch.tensor(np.array([]), device=self.device, dtype=torch.float32)
        
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
                    
                batch_preds = torch.sigmoid(batch_preds)
                epoch_loss += batch_loss.item()

                # flat predictions and masks
                batch_size = batch_img.shape[0]
                w = batch_img.shape[2]
                h = batch_img.shape[3]
                
                # 4 classes - 4 rows for every image
                preds_flat = batch_preds.view((batch_size * self.num_classes, w, h))
                mask_flat = batch_mask.view((batch_size * self.num_classes, w, h))
                # binarization
                preds_flat = (preds_flat > self.pred_threshold).float()
                mask_flat = (mask_flat > 0.5).float()
                
                if phase == 'valid':
                    predictions = torch.cat((predictions, preds_flat))
                    ground_truth = torch.cat((ground_truth, mask_flat))
                
                # avg dice on batch
                dice_batch = dice_coeff_mean(preds_flat, mask_flat)
                logs['dice_batch'] = dice_batch
                epoch_dice.append(dice_batch)
                
                del batch_mask, batch_img, batch_preds, preds_flat, mask_flat         
                #torch.cuda.empty_cache()
                
                # save current batch loss value for output
                loss_logs = {self.loss.__name__: batch_loss.item()}
                logs.update(loss_logs) 
                s = self._format_logs(logs)
                iterator.set_postfix_str(s)
        
            if phase == 'valid':
                dice_mean = dice_coeff_mean(predictions, ground_truth)
                del predictions, ground_truth
            else:
                dice_mean = np.mean(epoch_dice)
                
        epoch_loss /= data_len
        self.scheduler.step(epoch_loss)
        torch.cuda.empty_cache()
        
        return epoch_loss, dice_mean
        
    def run_model(self):
        """
        Iterate through epochs with both train and validation pahses
        """
        cur_time = datetime.now().strftime("%H:%M:%S")
        
        _fix_seeds()
        
        for epoch in range(self.num_epochs):
            _fix_seeds()
            
            print(f"Starting epoch: {epoch} | time: {cur_time}")
            print('LR:',[pg['lr'] for pg in self.optimizer.param_groups])
            for phase in ['train', 'valid']:
                
                curr_loss, dice_mean = self.run_epoch(phase)
                self.loss_history[phase].append(curr_loss)
                self.dice_history[phase].append(dice_mean)
                
                if phase == 'valid':
                    print(f'Valid loss: {curr_loss}')
                    print(f'Valid dice mean: {dice_mean}')
                
            if self.best_loss > self.loss_history['valid'][-1]:
                self.best_loss = self.loss_history['valid'][-1]
                torch.save(self.model, self.path_to_save_model)
                print('*** Model saved! ***\n')
                
            cur_time = datetime.now().strftime("%H:%M:%S")
