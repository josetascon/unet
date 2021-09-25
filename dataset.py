import os
import random

import numpy as np
import torch
from skimage.io import imread
from SimpleITK import ReadImage, GetArrayFromImage
from torch.utils.data import Dataset

from utils import pad_image, pad_sample #crop_sample, resize_sample, normalize_volume

def listdir_fullpath(d: str, sort: bool = True):
    files = [os.path.join(d, f) for f in os.listdir(d)]
    if sort: files.sort()
    return list(filter(os.path.isfile, files))

class SimpleDataset(Dataset):
    in_channels = 1
    out_channels = 1

    def __init__(self, folder_image: str, folder_mask: str,
                image_size: int = 256, transform = None, preload = True):
        self.folder_image = folder_image
        self.folder_mask = folder_mask
        self.image_size = image_size
        self.transform = transform
        self.preload = preload

        self.files_images = listdir_fullpath(self.folder_image)
        self.num = len(self.files_images)

        # Disble mask, useful when predicting only
        self.enable_mask = True if folder_mask != None else False

        # if self.enable_mask:
        self.files_masks = listdir_fullpath(self.folder_mask)
        assert len(self.files_images) == len(self.files_masks), f'Number of images and masks is different {len(self.files_images)}: {len(self.files_masks)}'

        if self.preload:
            self.images = []
            self.masks = []
            for i, file in enumerate(self.files_images):
                self.images.append( self.read_image(file) )
                if self.enable_mask: self.masks.append( self.read_image( self.files_masks[i]) )

    def __len__(self):
        return self.num

    def read_image(self, file: str):
        return GetArrayFromImage( ReadImage(file) )

    def __getitem__(self, idx):
        # Read images
        # image = []
        # mask = []

        if self.preload:
            image = self.images[idx]
            if self.enable_mask: mask = self.masks[idx]
        else:
            image = self.read_image(self.files_images[idx])
            if self.enable_mask: mask = self.read_image(self.files_masks[idx])
        
        # image = self.read_image(self.files_images[idx])
        # mask = self.read_image(self.files_masks[idx])
        # image = imread(self.files_images[idx])
        # mask = imread(self.files_masks[idx])
        image, mask = pad_sample((image, mask))

        # image = pad_image(image)
        image = image[np.newaxis, ...]

        # if self.enable_mask:
        # mask = pad_image(mask)
        mask = mask[np.newaxis, ...]

        # Transform
        if self.transform is not None:
            # if self.enable_mask: 
            image, mask = self.transform((image, mask))
            # else: image = self.transform(image)

        # # Fix dimensions (channels, height, width)
        # image = image.transpose(2, 0, 1)
        # mask = mask.transpose(2, 0, 1)

        # Tensors
        image_tensor = torch.from_numpy(image.astype(np.float32))
        # if self.enable_mask: 
        mask_tensor = torch.from_numpy(mask.astype(np.float32))

        # return tensors
        # if self.enable_mask:
        return image_tensor, mask_tensor
        # return image_tensor
