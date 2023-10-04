import torch
import torch.nn.functional as F
from tqdm import tqdm
import custom_metrics
import custom_plots

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from torcheval.metrics.functional import binary_auroc, binary_accuracy, binary_recall, binary_precision, binary_f1_score  
import wandb
import matplotlib.pyplot as plt



@torch.inference_mode()
def evaluate(net, dataloader, device, amp, experiment=None, logging_data=None):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    accuracy_score = 0
    precision_score = 0
    recall_score = 0    
    f1_score = 0
    auc_score = 0
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            #val_images, val_true_masks
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # val_pred
            # predict the mask
            mask_pred = net(image)

            #torch.Size([8, 1, 600, 600])                                                                                 
            #print( mask_pred.size() )
            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
                ### IN the binary case, it looks like it is more efficient to work directly weith the logits. Thus we apply the nonlinearity at the end, but this seems really subtle for such an imporrtant elemnt.
                # compute the Dice score
                #print( mask_pred.size() )
                #print( mask_true.size() )

                # menasche added squeeze(1)
                dice_score += dice_coeff(mask_pred.squeeze(1), mask_true, reduce_batch_first=False)
                bs = mask_true.shape[0]
                accuracy_score += binary_accuracy( mask_pred.squeeze(1).flatten(), mask_true.flatten() ) 
                precision_score += binary_precision( mask_pred.squeeze(1).flatten() , mask_true.flatten() ) 
                recall_score += binary_recall( mask_pred.squeeze(1).flatten() , mask_true.flatten() ) 
                f1_score += binary_f1_score( mask_pred.squeeze(1).flatten() , mask_true.flatten() ) 
                auc_score += binary_auroc( mask_pred.squeeze(1).flatten() , mask_true.flatten() ) 
                
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()

    output = ( dice_score / max(num_val_batches, 1), accuracy_score / max(num_val_batches, 1),
               precision_score / max(num_val_batches, 1), recall_score / max(num_val_batches, 1),
               f1_score / max(num_val_batches, 1), auc_score/ max(num_val_batches, 1) )            
    
    if experiment:
        logging_data["validation Dice"] = output[0]
        logging_data["validation accuracy"] = output[1]
        logging_data["validation precision"] = output[2]
        logging_data["validation recall"] = output[3]
        logging_data["validation f1"] = output[4]
        logging_data["validation auc"] = output[5]           
        logging_data["validation diagnostic"]["mask"] = wandb.Image( mask_true[0].float().cpu() )
        logging_data["validation diagnostic"]["prediction"] = wandb.Image( mask_pred[0].float().cpu() )
        logging_data["validation diagnostic"]["image"] = wandb.Image( image[0].float().cpu() )
        
        mask_true_cpy = mask_true.clone().unsqueeze(1)
        TP = torch.where( (mask_true_cpy == 1) & (mask_pred == 1));
        TN = torch.where( (mask_true_cpy == 0) & (mask_pred == 0));
        FP = torch.where( (mask_true_cpy == 0) & (mask_pred == 1));
        FN = torch.where( (mask_true_cpy == 1) & (mask_pred == 0));

        # mask true size: torch.Size([1, 600, 600])
        # mask pred size: torch.Size([1, 1, 600, 600])
        # TP size: 4
        #(tensor([0, 0, 0,  ..., 0, 0, 0], device='cuda:0'), tensor([0, 0, 0,  ..., 0, 0, 0], device='cuda:0'), tensor([ 50,  50,  51,  ..., 551, 551, 551], device='cuda:0'), tensor([333, 334, 320,  ..., 149, 150, 183], device='cuda:0'))

        TP_img = torch.zeros_like(mask_true_cpy)
        TP_img[TP] = 1;
        FP_img = torch.zeros_like(mask_true_cpy) 
        FP_img[FP] = 1;
        FN_img = torch.zeros_like(mask_true_cpy)
        FN_img[FN] = 1;

        _size = list( mask_true_cpy.shape )
        _size[1] = 3
        stoplight = torch.zeros( _size ,dtype=torch.long)
        green = torch.tensor([0,255,0])
        yellow = torch.tensor([255,255,0])
        red = torch.tensor([255,0,0])
        stoplight[TP[0],:,TP[2],TP[3]] = green
        stoplight[FN[0],:,FN[2],FN[3]] = yellow
        stoplight[FP[0],:,FP[2],FP[3]] = red

        fig = plt.figure(figsize=(7,7))
        ax = fig.gca()
        ax.imshow( image[0].permute(1,2,0).cpu() )
        ax.imshow( stoplight[0].permute(1,2,0).cpu() , alpha=0.45 )
        
        logging_data['validation diagnostic']["stoplight-overlay"] = fig            
        logging_data['validation diagnostic']["TP"] = wandb.Image( TP_img[0].float().cpu() )
        logging_data['validation diagnostic']["FP"] = wandb.Image( FP_img[0].float().cpu() )
        logging_data['validation diagnostic']["FN"] = wandb.Image( FN_img[0].float().cpu() )
        logging_data['validation diagnostic']["stoplight"] = wandb.Image( stoplight[0].float().cpu() )
        plt.close()

        ### 
        #statistic, p_value, pred, labels_out_pred  = custom_metrics.grain_size_distribution_metric( mask_true_cpy[0] , mask_pred[0] ,image[0] )
        #logging_data['validation diagnostic']["grain_size_distribution_stat"] = statistic
        #logging_data['validation diagnostic']["grain_size_distribution_stat"] = p_value

        try:
            experiment.log( logging_data )
            print("Successfully logged exp" )
        except:
            pass
        
    return output


@torch.inference_mode()
def evaluate_with_grainstats(net, dataloader, device, amp, experiment=None, logging_data=None):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    accuracy_score = 0
    precision_score = 0
    recall_score = 0    
    f1_score = 0
    auc_score = 0
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            #val_images, val_true_masks
            image, mask_true, grayscale = batch['image'], batch['mask'], batch['gray']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # val_pred
            # predict the mask
            mask_pred = net(image)

            #torch.Size([8, 1, 600, 600])                                                                                 
            #print( mask_pred.size() )
            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
                ### IN the binary case, it looks like it is more efficient to work directly weith the logits. Thus we apply the nonlinearity at the end, but this seems really subtle for such an imporrtant elemnt.
                # compute the Dice score
                #print( mask_pred.size() )
                #print( mask_true.size() )

                # menasche added squeeze(1)
                dice_score += dice_coeff(mask_pred.squeeze(1), mask_true, reduce_batch_first=False)
                bs = mask_true.shape[0]
                accuracy_score += binary_accuracy( mask_pred.squeeze(1).flatten(), mask_true.flatten() ) 
                precision_score += binary_precision( mask_pred.squeeze(1).flatten() , mask_true.flatten() ) 
                recall_score += binary_recall( mask_pred.squeeze(1).flatten() , mask_true.flatten() ) 
                f1_score += binary_f1_score( mask_pred.squeeze(1).flatten() , mask_true.flatten() ) 
                auc_score += binary_auroc( mask_pred.squeeze(1).flatten() , mask_true.flatten() ) 
                
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()

    output = ( dice_score / max(num_val_batches, 1), accuracy_score / max(num_val_batches, 1),
               precision_score / max(num_val_batches, 1), recall_score / max(num_val_batches, 1),
               f1_score / max(num_val_batches, 1), auc_score/ max(num_val_batches, 1) )            
    
    if experiment:
        logging_data["validation Dice"] = output[0]
        logging_data["validation accuracy"] = output[1]
        logging_data["validation precision"] = output[2]
        logging_data["validation recall"] = output[3]
        logging_data["validation f1"] = output[4]
        logging_data["validation auc"] = output[5]           
        logging_data["validation diagnostic"]["mask"] = wandb.Image( mask_true[0].float().cpu() )
        logging_data["validation diagnostic"]["prediction"] = wandb.Image( mask_pred[0].float().cpu() )
        logging_data["validation diagnostic"]["image"] = wandb.Image( image[0].float().cpu() )
        
        mask_true_cpy = mask_true.clone().unsqueeze(1)
        TP = torch.where( (mask_true_cpy == 1) & (mask_pred == 1));
        TN = torch.where( (mask_true_cpy == 0) & (mask_pred == 0));
        FP = torch.where( (mask_true_cpy == 0) & (mask_pred == 1));
        FN = torch.where( (mask_true_cpy == 1) & (mask_pred == 0));

        # mask true size: torch.Size([1, 600, 600])
        # mask pred size: torch.Size([1, 1, 600, 600])
        # TP size: 4
        #(tensor([0, 0, 0,  ..., 0, 0, 0], device='cuda:0'), tensor([0, 0, 0,  ..., 0, 0, 0], device='cuda:0'), tensor([ 50,  50,  51,  ..., 551, 551, 551], device='cuda:0'), tensor([333, 334, 320,  ..., 149, 150, 183], device='cuda:0'))

        TP_img = torch.zeros_like(mask_true_cpy)
        TP_img[TP] = 1;
        FP_img = torch.zeros_like(mask_true_cpy) 
        FP_img[FP] = 1;
        FN_img = torch.zeros_like(mask_true_cpy)
        FN_img[FN] = 1;

        _size = list( mask_true_cpy.shape )
        _size[1] = 3
        stoplight = torch.zeros( _size ,dtype=torch.long)
        green = torch.tensor([0,255,0])
        yellow = torch.tensor([255,255,0])
        red = torch.tensor([255,0,0])
        stoplight[TP[0],:,TP[2],TP[3]] = green
        stoplight[FN[0],:,FN[2],FN[3]] = yellow
        stoplight[FP[0],:,FP[2],FP[3]] = red

        fig = plt.figure(figsize=(7,7))
        ax = fig.gca()
        ax.imshow( image[0].permute(1,2,0).cpu() )
        ax.imshow( stoplight[0].permute(1,2,0).cpu() , alpha=0.45 )
        
        logging_data['validation diagnostic']["stoplight-overlay"] = fig            
        logging_data['validation diagnostic']["TP"] = wandb.Image( TP_img[0].float().cpu() )
        logging_data['validation diagnostic']["FP"] = wandb.Image( FP_img[0].float().cpu() )
        logging_data['validation diagnostic']["FN"] = wandb.Image( FN_img[0].float().cpu() )
        logging_data['validation diagnostic']["stoplight"] = wandb.Image( stoplight[0].float().cpu() )
        plt.close()

        ### 
        statistic, p_value, pred, labels_out_pred  = custom_metrics.grain_size_distribution_metric( mask_true_cpy[0] , mask_pred[0] , grayscale[0] )
        logging_data['validation grain_size_distribution_stat'] = statistic
        logging_data['validation grain_size_distribution_p_value'] = p_value

        fig = custom_plots.show_components(pred.cpu().numpy().squeeze(), labels_out_pred.cpu().squeeze())   
        logging_data['validation diagnostic']['connected components'] = fig
        
        try:
            experiment.log( logging_data )
            print("Successfully logged exp" )
        except:
            pass
        
    return output


