import os
import random

import numpy as np
import torch
from skimage.io import imread
import SimpleITK as sitk
from torch.utils.data import Dataset

from utils import pad_image, pad_sample, normalize_volume, gaussian_noise #crop_sample, resize_sample

def listdir_fullpath(d: str, sort: bool = True):
    files = [os.path.join(d, f) for f in os.listdir(d)]
    if sort: files.sort()
    return list(filter(os.path.isfile, files))

class SimpleDataset(Dataset):
    in_channels = 1
    out_channels = 1

    def __init__(self, folder_image: str, folder_mask: str,
                image_size: int = 256, num_images: int = -1, 
                transform = None, preload = True):
        self.folder_image = folder_image
        self.folder_mask = folder_mask
        self.image_size = image_size
        self.transform = transform
        self.preload = preload

        self.files_images = listdir_fullpath(self.folder_image)
        if num_images > 0: self.files_images = self.files_images[:num_images]
        self.num = len(self.files_images)
        print('Create Dataset with {} images'.format(self.num))

        # Disble mask, useful when predicting only
        self.enable_mask = True if folder_mask != None else False

        if self.enable_mask:
            self.files_masks = listdir_fullpath(self.folder_mask)
            if num_images > 0: self.files_masks = self.files_masks[:num_images]
            assert len(self.files_images) == len(self.files_masks), f'Number of images and masks is different {len(self.files_images)}: {len(self.files_masks)}'

        # Load images during initialization
        if self.preload:
            self.images = []
            self.masks = []
            for i, file in enumerate(self.files_images):
                self.images.append( self.read_image(file) )
                if self.enable_mask: self.masks.append( self.read_image( self.files_masks[i]) )

        self.padding = self.compute_padding()

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        # Read images
        image = []
        mask = []

        if self.preload:
            # Get preloaded images
            image = self.images[idx]
            if self.enable_mask: mask = self.masks[idx]
        else:
            # Read and get image
            image = self.read_image(self.files_images[idx])
            if self.enable_mask: mask = self.read_image(self.files_masks[idx])

        image = pad_image(image)
        if self.enable_mask:
            mask = pad_image(mask)

        # Transform
        if self.transform is not None:
            if self.enable_mask:
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]
            else: 
                transformed = self.transform(image=image)
                image = transformed['image']

        # # Fix dimensions (channels, height, width)
        # image = image.transpose(2, 0, 1)
        # mask = mask.transpose(2, 0, 1)

        if not self.enable_mask:
            image = normalize_volume(image)

        image = image[np.newaxis, ...]
        if self.enable_mask: mask = mask[np.newaxis, ...]

        # Tensors
        image_tensor = torch.from_numpy(image.astype(np.float32))
        if self.enable_mask: mask_tensor = torch.from_numpy(mask.astype(np.float32))

        # return tensors
        if self.enable_mask:
            return image_tensor, mask_tensor
        return image_tensor

    # def set_transform(self, trfm):
    #     self.transform = trfm

    def read_image(self, file: str):
        return sitk.GetArrayFromImage( sitk.ReadImage(file) )

    def reference_image(self):
        return sitk.ReadImage( self.files_images[0] )

    def compute_padding(self):
        image = self.read_image( self.files_images[0] )
        a = image.shape[0]
        b = image.shape[1]        
        level = 16
    
        if a%level == 0 and b%level == 0:
            return ((0,0),(0,0))
        
        if a%level != 0:
            diff = (level - a%level) / 2.0
            w = (int(np.floor(diff)), int(np.ceil(diff)))
        else:
            w = (0,0)
        
        if b%level != 0:
            diff = (level - b%level) / 2.0
            h = (int(np.floor(diff)), int(np.ceil(diff)))
        else:
            h = (0,0)
        padding = (w,h)
        return padding

class SubsetDataset(Dataset):
    def __init__(self, subset, transform=None, noise = False):
        self.subset = subset
        self.transform = transform
        self.noise = noise
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        xnp = x.numpy()
        ynp = y.numpy()
        if self.transform is not None:
            for k in range(xnp.shape[0]):
                # Spatial transformations
                transformed = self.transform(image=xnp[k,:,:], mask=ynp[k,:,:])
                xnp[k,:,:] = transformed["image"]
                ynp[k,:,:] = transformed["mask"]

        # Pixel intensity transformations
        xnp = normalize_volume( xnp )
        if self.noise:
            xnp = gaussian_noise( xnp, 0.2 )
        
        x = torch.from_numpy( xnp )
        y = torch.from_numpy( ynp )
        return x, y
        
    def __len__(self):
        return len(self.subset)