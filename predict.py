import os
import argparse

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader

from dataset import SimpleDataset
from unet import UNet
from utils import unpad_image


def main():
    args = parse_arguments()
    makedirs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    # Dataset
    folder_image = os.path.join(args.input, 'image')
    dataset = SimpleDataset(folder_image, folder_mask=None, num_images = args.num_images)

    # Create data loader
    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=False)
    loader = DataLoader(dataset, shuffle=False, **loader_args)

    # Print
    print('Load weights from file:', args.weights)
    
    # Read unet and model
    with torch.set_grad_enabled(False):
        unet = UNet(in_channels=SimpleDataset.in_channels, out_channels=SimpleDataset.out_channels)
        state_dict = torch.load(args.weights, map_location=device)
        unet.load_state_dict(state_dict)
        unet.eval()
        unet.to(device)

        for i, data in enumerate(loader):
            x = data
            x = x.to(device)

            y_pred = unet(x)
            y_pred_np = y_pred.detach().cpu().numpy()
            
            y_pred_np = y_pred_np[0,0,:,:]
            y_pred_np = unpad_image(y_pred_np, dataset.padding)
            # print(y_pred_np.shape)

            file_output = args.output + '_{:04d}.{}'.format(i,args.format)
            image_ref = dataset.reference_image()
            image_out = sitk.GetImageFromArray(y_pred_np)
            image_out.CopyInformation(image_ref)
            sitk.WriteImage( image_out, file_output  )
            print('Write image:', file_output)

    return

def makedirs(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Precit segmentation with U-Net model')
    parser.add_argument('--input', '-i', type=str, default='./data/', 
        help='Input path with image and mask folder')
    parser.add_argument('--output', '-o', type=str, default='./out/mask', 
        help='Output path folder and prefix')
    parser.add_argument('--format', type=str, default='nii', 
        help='Output format of files')
    parser.add_argument('--num_images', type=int, default=-1,
        help='Number of images to include from data folders. If < 1 include all. Default -1.')
    
    parser.add_argument('--batch-size', type=int, default=1,
        help='Batch size for training (default: 1)')
    parser.add_argument('--device', type=str, default='cuda:0',
        help='device for training (default: cuda:0)')
    parser.add_argument('--workers', type=int, default=4,
        help='number of workers for data loading (default: 4)')
    parser.add_argument('--weights', type=str, default='./weights/unet.pt', 
        help='folder with weights')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()