import os
import random
import pickle

import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from configs import (
    TRAIN_PATH,
    TEST_PATH,
    CLASS_RGB_COLOR,
    SEED
)


def _fix_seeds(seed=SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def mask2rle(img):
    """
    https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode    
    
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    
    return ' '.join(str(x) for x in runs)


def rle2mask(mask_rle, shape=(1600,256)):
    """
    https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
    
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
        
    return img.reshape(shape).T


def plot_loss_metrics(loss_history, metrics_history):
    """
    loss_history: dict of lists, contains train and valid loss
    metrics_history: dict of 2 dicts of lists, metrics on train and validation
    """
    fig, ax = plt.subplots(nrows=len(metrics_history) + 1, ncols=1, figsize=(10, 6 * len(metrics_history) + 1))
    ax[0].set_title('Loss')
    ax[0].plot(loss_history['train'], label='train')
    ax[0].plot(loss_history['valid'], label='valid')
    ax[0].legend()
    
    for ind, metric in enumerate(metrics_history):
        ax[ind + 1].set_title(metric)
        ax[ind + 1].plot(metrics_history[metric]['train'], label='train')
        ax[ind + 1].plot(metrics_history[metric]['valid'], label='valid')
        ax[ind + 1].legend()

    plt.show()

    
def plot_batch_metrics(trainer):
    """
    from trainer get metrics_history: dict (train or validation) of list (epochs) of list (metrics)
    """
    metric_names = {
        0: 'dices_batch',
        1: 'cnt_neg_pred',
        2: 'cnt_neg_mask',
        3: 'cnt_pos_pred',
        4: 'cnt_pos_mask',
        5: 'precision_batch',
        6: 'recall_batch',
    }
    metrics_history = {}

    for metric in range(len(metric_names)):
        metrics_history[metric_names[metric]] = {
            'train': [],
            'valid': [],
        }
        for epoch in range(trainer.num_epochs):
            metrics_history[metric_names[metric]]['train'].extend(
                trainer.epoch_additional_metrics['train'][epoch][metric])
            metrics_history[metric_names[metric]]['valid'].extend(
                trainer.epoch_additional_metrics['valid'][epoch][metric])
    
    fig, ax = plt.subplots(nrows=len(metrics_history), ncols=2, figsize=(16, 5 * len(metrics_history)))
    for ind, metric in enumerate(metrics_history):
        
        ax[ind][0].set_title(metric)
        ax[ind][1].set_title(metric)
        
        if metric in ['dices_batch', 'cnt_neg_pred', 'cnt_neg_mask']: 

            ax[ind][0].plot( metrics_history[metric]['train'], label='train', alpha=0.5)
            ax[ind][1].plot( metrics_history[metric]['valid'], label='valid', alpha=0.5)
        else:
            for cls in range(4):
                m_tr = metrics_history[metric]['train']
                m_vl = metrics_history[metric]['valid']
                xx_tr = list(range(len(np.array(m_tr)[:, 0])))
                xx_vl = list(range(len(np.array(m_vl)[:, 0])))
                ax[ind][0].scatter(xx_tr, np.array(m_tr)[:, cls], 
                                label='cls ' + str(cls + 1), alpha=0.5, s=20)
                ax[ind][1].scatter(xx_vl, np.array(m_vl)[:, cls], 
                                label='cls ' + str(cls + 1), alpha=0.5, s=20)            
        ax[ind][0].legend()
        ax[ind][1].legend()
        
    plt.show()
    
    
def show_images_with_defects(image_path,
                             images_dict,
                             with_mask=True,
                             random_sample=True,
                             seed=42,
                             sample_size=20, 
                             image_ids=None,
                             filled_regions=False,
                             cls_color=CLASS_RGB_COLOR):
    """
    Show images from dataset.
    
    Parameters:
        :param image_path: str, path to image directory
        :param images_dict: dict of dicts, rle for every class for every image
        :param with_mask: bool, show pictures with or without mask
        :param random_sample: bool, show random pictures or not
        :param sample_size: int, how many pictures to select randomly
        :param image_ids: list, image ids if `random_sample` = False
        :param seed: int, seed for random image taking
        :param filled_regions: show images with color filled regions
        :param cls_color: rgb tuples for class colors
    """
    if random_sample:
        random.seed(seed)
        image_ids = random.sample(os.listdir(image_path), sample_size)
    else:
        if image_ids is None:
            raise ValueError('Provide "image_ids" list')
            
    # show all images in "image_ids"
    image_ids_num = len(image_ids)
    fig, ax = plt.subplots(figsize=(15, 3 * image_ids_num), ncols=1, nrows=image_ids_num)
    
    for ind, img_id in enumerate(image_ids):
        # get image
        img = cv2.imread(image_path + img_id)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask_label = ''
        if with_mask:
            # for every class get mask, add to subplot
            mask_label += 'Classes: '
            for cls in range(1,5):
                cur_rle = images_dict[img_id][cls]
                if cur_rle != '':
                    mask_label += str(cls) 
                    mask_label += ' '
                    
                    mask = rle2mask(cur_rle)            
                    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

                    for i in range(0, len(contours)):
                        if filled_regions:
                            cv2.drawContours(img, contours=contours, contourIdx=i,
                                             color=cls_color[cls],
                                             thickness=cv2.FILLED)
                        else:
                            cv2.polylines(img, contours[i], True, cls_color[cls], 2)
                
        ax[ind].set_title(img_id + ' ' + mask_label)
        ax[ind].imshow(img)
        
    plt.tight_layout()  
    plt.show()
