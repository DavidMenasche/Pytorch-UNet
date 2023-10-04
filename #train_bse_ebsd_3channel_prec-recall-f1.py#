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
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset, BSE_EBSD_Dataset, BSE_EBSD_Dataset_3Channel
from utils.dice_score import dice_loss

import kornia
from kornia.augmentation import RandomRotation, RandomHorizontalFlip, RandomVerticalFlip

sys.path.append( "/home/appuser/data/TopoLoss/" )
import topoloss_pytorch

import numpy as np

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        _lambda: float = 1e-5,
        sOptimizer: str = 'rmsprop',
        sDir_img: str = './data/color-imgs/',
        alpha: float = 0.5,
):

    dir_img = Path(sDir_img)
    dir_mask = Path('./data/masks/')
    #_lambda = 1e-5
    
    # do this after we get set up.
    pretrain_epoch = epochs-3;

    # 1. Create dataset
    try:
        # dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
        dataset = BSE_EBSD_Dataset_3Channel( dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging) This is disabled during sweep
    
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')

    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
        Dataset: {sDir_img}
        Alpha: {alpha}
    ''')

    t=datetime.datetime.now();
    dir_checkpoint = Path('./{}_checkpoints_{}'.format( experiment.name , t.strftime("%Y-%m-%d-%H.%M.%S") ) )

    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)
    else:
        print('Abort before rewriting exp..')
        exit()


    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    if sOptimizer == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(),
                                  lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    elif sOptimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,foreach=True)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=epochs)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # menasche add the augmentor:

    augmentor = kornia.augmentation.AugmentationSequential( RandomRotation(degrees=45,p=1),
                                                            RandomHorizontalFlip(p=0.5),
                                                            RandomVerticalFlip(p=0.5),
                                                            data_keys=["image","mask"],
                                                            same_on_batch=False )

    print( "Num classes: {} ".format( model.n_classes ) )
    bDebug = False
    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                # menasche added augmentation
                out = augmentor( images, true_masks.type(torch.float32).reshape(-1,1, true_masks.size()[1], true_masks.size()[2]) )
                images = out[0]
                true_masks = out[1].reshape(-1,true_masks.size()[1],true_masks.size()[2] )
                
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        ## 'criterion' is the first term off the loss function. Looks like what was implemented was a combination loss
                        ## with 50% weight to each BCE and dice loss. 
                        loss = alpha*criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += (1-alpha)*dice_loss(F.sigmoid(masks_pred).squeeze(1), true_masks.float(), multiclass=False)
                        #if epoch > pretrain_epoch:
                        #    loss += _lambda*topoloss_pytorch.getTopoLoss( F.sigmoid(masks_pred.squeeze(1)) , true_masks.float(), topo_size=148 )                            
                    else:

                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        tmp = (torch.sigmoid( masks_pred[0,0,:,:].clone() ) > 1).float()
                        
                        logging_dict = {
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': None,
                            'images': wandb.Image(images[0].cpu()),
                            'training diagnostic': {
                                'mask': wandb.Image(true_masks[0].float().cpu()),
                                'prediction': wandb.Image( tmp.cpu() )       
                            },
                            'validation diagnostic': {
                                'mask': None, #placeholder wandb.Image( val_true_masks[0].float().cpu() ),
                                'prediction': None # wandb.Image( val_pred[0,0,:,:].float().cpu() )
                            },                                                         
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        }
                        
                        val_scores = evaluate(model, val_loader, device, amp, experiment=experiment, logging_data=logging_dict )
                        scheduler.step(val_scores[0])

                        logging.info('Validation Dice score: {}'.format(val_scores[0]))
                        logging.info('Validation Accuracy score: {}'.format(val_scores[1]))
                        logging.info('Validation Precision score: {}'.format(val_scores[2]))
                        logging.info('Validation Recall score: {}'.format(val_scores[3]))
                        logging.info('Validation F1 score: {}'.format(val_scores[4]))
                        logging.info('Validation AUC score: {}'.format(val_scores[5]))
                       
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


def sweepfcn2( config=None ):
    with wandb.init(config=config):
        config = wandb.config        
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {device}')
        
        # Change here to adapt to your data
        # n_channels=3 for RGB images
        # n_classes is the number of probabilities you want to get per pixel
        model = UNet(n_channels=3, n_classes=config.classes, bilinear=config.bilinear)
        model = model.to(memory_format=torch.channels_last)
        
        logging.info(f'Network:\n'
                     f'\t{model.n_channels} input channels\n'
                     f'\t{model.n_classes} output channels (classes)\n'
                     f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')
        
        if config.load:
            state_dict = torch.load(config.load, map_location=device)
            del state_dict['mask_values']
            model.load_state_dict(state_dict)
            logging.info(f'Model loaded from {config.load}')

        model.to(device=device)
        try:
            train_model(
                model=model,
                epochs=config.epochs,
                batch_size=config.batch_size,
                learning_rate=config.lr,
                device=device,
                img_scale=config.scale,
                val_percent=config.val / 100,
                amp=config.amp,
                _lambda=config._lambda,
                sOptimizer=config.sOptimizer,
                sDir_img=config.sDir_img,
                alpha=config.alpha
            )
        except torch.cuda.OutOfMemoryError:
            logging.error('Detected OutOfMemoryError! '
                          'Enabling checkpointing to reduce memory usage, but this slows down training. '
                          'Consider enabling AMP (--amp) for fast and memory efficient training')
            torch.cuda.empty_cache()
            model.use_checkpointing()
            train_model(
                model=model,
                epochs=config.epochs,
                batch_size=config.batch_size,
                learning_rate=config.lr,
                device=device,
                img_scale=config.scale,
                val_percent=config.val / 100,
                amp=config.amp,
                _lambda=config._lambda,
                sOptimizer=config.sOptimizer,
                sDir_img=config.sDir_img,
                alpha=config.alpha
            )

if __name__ == '__main__':

    par_dict = {
        'classes':{
            'value' : 1
        },
        'amp' : {
            'value' : True
        },
        'bilinear' : {
            'value' : False
        },
        'val' : {
            'value' : 10
        },
        'scale' : {
            'value' : 1.0
        },
        'load' : {
            'value' : False
        },
        'lr' : {
            'value' : 0.00076268
        },
        #'lr' : {
        #    'distribution' : 'uniform',
        #    'min' : 1e-7 ,
        #    'max' : 1e-3
        #},
        'batch_size' : {
            #'values' : [8,12,16,20]
            'value' : 8
        },
        'epochs' : {
            'value' : 20
        },
        '_lambda' : {
            #'distribution' : 'uniform',
            #'min' : 1e-7 ,
            #'max' : 1e-3            
            'value' : 0.0006649890 #0.0006649
        },
        'sOptimizer' : {
            'value' : 'adam'
        },
        'sDir_img' : {
            'value' : './data/color-imgs_serial_0/'
            #'values' : ['./data/color-imgs_serial_0/','./data/color-imgs_serial_1/','./data/color-imgs_serial_2/','./data/color-imgs_serial_3/','./data/color-imgs_serial_4/','./data/color-imgs_serial_5/','./data/color-imgs_serial_6/','./data/color-imgs_serial_7/','./data/color-imgs_serial_8/','./data/color-imgs_serial_9/']
        },
        'alpha' : {
            'distribution' : 'uniform',
            'min' : 0,
            'max' : 1
        }
    }

    sweep_configuration = {
        'method': 'random',
        'metric': 
        {
            'goal': 'minimize', 
            'name': 'loss'
        },
        'parameters': par_dict
    }
    import pprint
    pprint.pprint( sweep_configuration )
    print( "starting sweep" )
    # 3: Start the sweep
    sweep_id = wandb.sweep(
        sweep=sweep_configuration, 
        project='U-Net-BSE-3Channel-alphaStudy'
    )

    wandb.agent(sweep_id, function=sweepfcn2, count=10)


        
"""
if __name__ == '__main__':
    args = get_args()
    print( args )
    exit
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )


"""
