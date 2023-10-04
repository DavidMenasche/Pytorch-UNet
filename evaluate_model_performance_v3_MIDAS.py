import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import datetime

import wandb
from evaluate import evaluate_with_grainstats
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset, BSE_EBSD_Dataset, BSE_EBSD_Dataset_3Channel, BSE_EBSD_Dataset_3Channel_and_Gray
from utils.dice_score import dice_loss

import kornia
from kornia.augmentation import RandomRotation, RandomHorizontalFlip, RandomVerticalFlip

#sys.path.append( "/home/a#ppuser/data/TopoLoss/" )
#import topoloss_pytorch

import numpy as np

for idx in range(1):

    img_scale = 1;
    val_percent=0.1
    model_to_eval = '/home/appuser/data/torch/Pytorch-UNet/icy-sweep-3_checkpoints_2023-09-07-19.57.32/checkpoint_epoch20.pth'
    dir_img = Path('./data/happy-trails-color-imgs-test_serial_{}/'.format(idx))
    dir_mask = Path('./data/happy-trails-test-masks/')
    dir_grayscale = Path('./data/happy-trails-grayscale-imgs-test_serial_0/')
    amp=True
    
    dataset = BSE_EBSD_Dataset_3Channel_and_Gray( dir_img, dir_mask, dir_grayscale, img_scale)
    #n_val = int(len(dataset) * val_percent)
    #n_train = len(dataset) - n_val
    #train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    batch_size=1
    
    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    #val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    val_loader = DataLoader(dataset, shuffle=False, drop_last=True, **loader_args)
    
    # (Initialize logging) This is disabled during sweep
    
    experiment = wandb.init(project='fresh_testing', resume='allow', anonymous='must')
    
    #experiment.config.update(
    #    dict( batch_size=batch_size, img_scale=img_scale, amp=amp)
    #)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    model = UNet(n_channels=3, n_classes=1, bilinear=False)
    model = model.to(memory_format=torch.channels_last)
    
    state_dict = torch.load(model_to_eval, map_location=device)
    del state_dict['mask_values']
    model.load_state_dict(state_dict)
    logging.info(f'Model loaded from {model_to_eval}')
    model.to(device=device)
    
    logging_dict = {
        'validation Dice': None,
        'validation diagnostic': {
            'mask': None, #placeholder wandb.Image( val_true_masks[0].float().cpu() ),
            'prediction': None # wandb.Image( val_pred[0,0,:,:].float().c
        }
    }
    
    val_scores = evaluate_with_grainstats(model, val_loader, device, amp, experiment=experiment, logging_data=logging_dict )
    
    logging.info('Validation Dice score: {}'.format(val_scores[0]))
    logging.info('Validation Accuracy score: {}'.format(val_scores[1]))
    logging.info('Validation Precision score: {}'.format(val_scores[2]))
    logging.info('Validation Recall score: {}'.format(val_scores[3]))
    logging.info('Validation F1 score: {}'.format(val_scores[4]))
    logging.info('Validation AUC score: {}'.format(val_scores[5]))
    
