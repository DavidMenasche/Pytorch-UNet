from pathlib import Path
import data_loading
from torchvision.io import read_image
import torch.nn as nn


def testing():
    dir_img = Path('./data/imgs/')
    dir_mask = Path('./data/masks/')
    dir_checkpoint = Path('./checkpoints/')    
    dataset = data_loading.BSE_EBSD_Dataset(dir_img, dir_mask, img_scale)
    
if __name__ == "__main__":
    testing()
