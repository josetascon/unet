import os
import random

import numpy as np
import torch
from skimage.io import imread
from torch.utils.data import Dataset

from utils import crop_sample, pad_sample, resize_sample, normalize_volume

def listdir_fullpath(d: str, sort: bool = True):
    files = [os.path.join(d, f) for f in os.listdir(d)]
    if sort: files.sort()
    return list(filter(os.path.isfile, files))

class SimpleDataset(Dataset):
    in_channels = 1
    out_channels = 1

    def __init__(self, folder_image: str, folder_mask: str,
                image_size: int = 256, transform = None):
        self.folder_image = folder_image
        self.folder_mask = folder_mask
        self.image_size = image_size
        self.transform = transform

        self.files_images = listdir_fullpath(self.folder_image)
        self.files_masks = listdir_fullpath(self.folder_mask)

        self.num = len(self.files_images)
        assert len(self.files_images) == len(self.files_masks), f'Number of images and masks is different {len(self.files_images)}: {len(self.files_masks)}'

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        # Read images
        image = imread(self.files_images[idx])
        mask = imread(self.files_masks[idx])

        image, mask = pad_sample((image, mask))

        image = image[np.newaxis, ...]
        mask = mask[np.newaxis, ...]

        # Transform
        if self.transform is not None:
            image, mask = self.transform((image, mask))

        # # Fix dimensions (channels, height, width)
        # image = image.transpose(2, 0, 1)
        # mask = mask.transpose(2, 0, 1)

        # Tensors
        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))

        # return tensors
        return image_tensor, mask_tensor
