import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SimpleDataset
from unet import UNet
from utils import dsc


def main():
    args = parse_arguments()
    makedirs(args)
    snapshotargs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    # Dataset
    folder_image = os.path.join(args.input, 'images')
    folder_mask = os.path.join(args.input, 'masks')
    dataset = SimpleDataset(folder_image, folder_mask, args.image_size)

    # Create data loader
    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=False)
    loader = DataLoader(dataset, shuffle=True, **loader_args)
    
    # Read unet and model
    with torch.set_grad_enabled(False):
        unet = UNet(in_channels=SimpleDataset.in_channels, out_channels=SimpleDataset.out_channels)
        state_dict = torch.load(args.weights, map_location=device)
        unet.load_state_dict(state_dict)
        unet.eval()
        unet.to(device)

        input_list = []
        pred_list = []
        true_list = []

        for i, data in tqdm(enumerate(loader)):
            x, y_true = data
            x, y_true = x.to(device), y_true.to(device)

            y_pred = unet(x)
            y_pred_np = y_pred.detach().cpu().numpy()
            pred_list.extend([y_pred_np[s] for s in range(y_pred_np.shape[0])])

            y_true_np = y_true.detach().cpu().numpy()
            true_list.extend([y_true_np[s] for s in range(y_true_np.shape[0])])

            x_np = x.detach().cpu().numpy()
            input_list.extend([x_np[s] for s in range(x_np.shape[0])])

    

def makedirs(args):
    os.makedirs(args.predictions, exist_ok=True)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Precit segmentation with U-Net model')
    parser.add_argument('--input', '-i', type=str, default='./data/', 
        help='Input path with image and mask folder')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=0.2,
        help='Percent of the data that is used as validation (0-1)')
    
    parser.add_argument('--batch-size', type=int, default=2,
        help='Batch size for training (default: 2)')
    parser.add_argument('--epochs', type=int, default=20,
        help='number of epochs to train (default: 20)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
        help='initial learning rate (default: 0.001)')
    parser.add_argument('--device', type=str, default='cuda:0',
        help='device for training (default: cuda:0)')
    parser.add_argument('--workers', type=int, default=4,
        help='number of workers for data loading (default: 4)')
    parser.add_argument('--vis-images', type=int, default=200,
        help='number of visualization images to save in log file (default: 200)')
    parser.add_argument('--vis-freq', type=int, default=10,
        help='frequency of saving images to log file (default: 10)')
    parser.add_argument('--weights', type=str, default='./weights', 
        help='folder to save weights')
    parser.add_argument('--logs', type=str, default='./logs', 
        help='folder to save logs')
    
    parser.add_argument('--image-size', type=int, default=256,
        help='target input image size (default: 256)')
    parser.add_argument('--aug-scale', type=int, default=0.05,
        help='scale factor range for augmentation (default: 0.05)')
    parser.add_argument('--aug-angle', type=int, default=10,
        help='rotation angle range in degrees for augmentation (default: 10)')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()

