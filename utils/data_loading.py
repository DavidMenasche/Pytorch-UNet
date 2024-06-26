import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

# menasche add
import cv2
import torchvision.transforms as T
import albumentations as A

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    # next two return cv2 images. This is inconsistency is bad practice
    elif ext == '.tif':
        return cv2.imread( str(filename),-1)
    elif ext == '.png':
        return cv2.imread( str(filename) ,-1)
        #return cv2.imread( str(filename) )
    else:
        return Image.open(filename)

def load_color_image(filename):
    ext = splitext(filename)[1]
    if ext == '.png':
        return cv2.imread( str(filename) )
    else:
        return Image.open(filename)

def load_mask( filename ):
    ext = splitext(filename)[1]
    if ext == '.png':
        return cv2.imread( str(filename) ,-1)
    

def unique_mask_values(idx, mask_dir, mask_suffix):
    #print(list(mask_dir.glob(idx + mask_suffix + '.*'))[0])
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '', gray_dir=None):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        #assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        if gray_dir:
            self.gray_dir=gray_dir
        
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')

###################################
        
class BSE_EBSD_Dataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')

    @staticmethod
    def preprocess(mask_values, cv2_img, scale, is_mask):
        w, h = cv2_img.shape
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'

        #pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        if is_mask: 
            cv2_img = cv2.resize( cv2_img, (newW,newH), cv2.INTER_NEAREST )
        else:
            cv2_img = cv2.resize( cv2_img , (newW, newH), cv2.INTER_LINEAR )
     
        img = np.asarray(cv2_img,dtype=np.float32)            
        #img = np.asarray(cv2_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i
            return mask
        
        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / ( 65535.00 )

            return img

    def __getitem__(self, idx ):
            
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class BSE_EBSD_Dataset_3Channel(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')

    @staticmethod
    def preprocess(mask_values, cv2_img, scale, is_mask):
        if is_mask:
            w, h = cv2_img.shape
        else:
            w, h, d = cv2_img.shape
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'

        #pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        if is_mask: 
            cv2_img = cv2.resize( cv2_img, (newW,newH), cv2.INTER_NEAREST )
        else:
            cv2_img = cv2.resize( cv2_img , (newW, newH), cv2.INTER_LINEAR )
     
        img = np.asarray(cv2_img,dtype=np.float32)            
        #img = np.asarray(cv2_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i
            return mask
        
        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / ( 255 )

            return img

    def __getitem__(self, idx ):
            
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_mask(mask_file[0])
        img = load_color_image(img_file[0])

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }



class BSE_EBSD_Dataset_3Channel_and_Gray(BasicDataset):
    def __init__(self, images_dir, mask_dir, gray_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask', gray_dir=gray_dir)

    @staticmethod
    def preprocess(mask_values, cv2_img, scale, is_mask , is_gray):
        if is_mask or is_gray:
            w, h = cv2_img.shape
        else:
            w, h, d = cv2_img.shape
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'

        if is_mask: 
            cv2_img = cv2.resize( cv2_img, (newW,newH), cv2.INTER_NEAREST )
        else:
            cv2_img = cv2.resize( cv2_img , (newW, newH), cv2.INTER_LINEAR )
     
        img = np.asarray(cv2_img,dtype=np.float32)            

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i
            return mask
        
        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if is_gray:
                if (img > 1).any():
                    img = img / ( 65535 )
            else:
                if (img > 1).any():
                    img = img / ( 255 )

            return img

    def __getitem__(self, idx ):
            
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))
        gray_file = list(self.gray_dir.glob(name + ".*"))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(gray_file) == 1, f'Either no grayscale or multiple grayscales found for the ID {name}: {gray_file}'

        mask = load_mask(mask_file[0])
        img = load_color_image(img_file[0])
        gray = load_image(gray_file[0])

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False, is_gray=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True, is_gray=False)
        gray = self.preprocess(self.mask_values, gray, self.scale, is_mask=False,is_gray=True)
        
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'gray': torch.as_tensor(gray.copy()).float().contiguous()
        }
