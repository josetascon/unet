import os
import argparse
import logging
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import SimpleDataset, SubsetDataset
from loss import DiceLoss
from unet import UNet
from utils import log_images, dsc
from transform import train_transform

import matplotlib.pyplot as plt

def main():
    args = parse_arguments()

    # Check if a model exist
    if os.path.exists(args.weights) and not args.overwrite:
        print('Existing file {}. Use option -w to overwrite model.'.format(args.weights))
        return

    makedirs(args)
    # snapshotargs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    # Dataset
    folder_image = os.path.join(args.input, args.image)
    folder_mask = os.path.join(args.input, args.mask)

    dataset = SimpleDataset(folder_image, folder_mask, args.image_size, args.num_images)
    assert len(dataset) > 0, f'Dataset lenght should be greater than zero'
    
    # Split into train / validation partitions
    n_val = int(len(dataset) * args.val)
    n_train = len(dataset) - n_val
    train_subset, val_subset = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # Transform in train_set
    noise = not args.disable_noise
    trfm = train_transform()
    if args.disable_augment:
        trfm = None
        noise = False
    train_set = SubsetDataset(train_subset, trfm, noise = noise)
    val_set = SubsetDataset(val_subset, noise = noise)

    if len(dataset) == 1: val_set = SubsetDataset(train_subset, trfm, noise = noise)

    # Create data loaders
    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=False)
    loader_train = DataLoader(train_set, shuffle=True, **loader_args)
    loader_valid = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    # loaders = {"train": loader_train, "valid": loader_valid}

    # Log file
    logging.basicConfig(filename=args.log, level=logging.INFO, 
        filemode='w', encoding='utf-8', format='%(levelname)s: %(message)s')
    logging.info("Arguments:")
    logging.info(vars(args))
    logging.info(f'''Starting training:
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Learning rate:   {args.learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}''')

    assert args.batch_size <= len(train_set), f'Batch size is greater than train set size'
    assert args.batch_size <= len(val_set), f'Batch size is greater than validation set size'

    unet = UNet(in_channels=SimpleDataset.in_channels, out_channels=SimpleDataset.out_channels)
    unet.to(device)

    dsc_loss = DiceLoss()
    best_validation_dice = 0.0
    best_epoch = 0

    optimizer = optim.Adam(unet.parameters(), lr=args.learning_rate)

    if args.plot:
        plt.figure()

    for epoch in tqdm(range(args.epochs), total=args.epochs):
        print()
        train(loader_train, unet, dsc_loss, optimizer, device, epoch)
        if epoch%args.validation_step == 0:
            dice = validate(loader_valid, unet, dsc_loss, device, epoch, args)

        if dice > best_validation_dice:
            best_epoch = epoch
            best_validation_dice = dice
            torch.save(unet.state_dict(), args.weights )

    text_best = "Best validation dice: {:4f} (epoch {})".format(best_validation_dice, best_epoch)
    text_output = 'Store weights to file: {}\n'.format(args.weights)
    logging.info(text_best)
    logging.info(text_output)
    print(text_best)
    print(text_output)

    return

def train(loader_train, model, criterion, optimizer, device, epoch):
    loss_train = []
    model.train()
    stream = tqdm(loader_train)
    with torch.set_grad_enabled(True):
        for i, (x, y_true) in enumerate(stream, start=1):
            x, y_true = x.to(device), y_true.to(device)
            y_pred = model(x)
            # print('input', x.shape)
            # print('pred:', y_pred.shape)
            # print('true:',y_true.shape)

            loss = criterion(y_pred, y_true)
            loss_train.append( 1.0 - loss.item() )
            dice = np.mean(loss_train)
            text = 'Epoch: {}. Training.   Loss: {:4.5f}'.format(epoch, dice)
            stream.set_description(text)
            logging.info(text)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # plot_epoch(x, y_true, y_pred)
    return

def validate(loader_valid, model, criterion, device, epoch, args):
    loss_valid = []
    num = len(loader_valid)
    model.eval()
    stream = tqdm(loader_valid)
    with torch.set_grad_enabled(False):
        for i, (x, y_true) in enumerate(stream, start=1):
            x, y_true = x.to(device), y_true.to(device)
            y_pred = model(x)

            loss = criterion(y_pred, y_true)
            loss_valid.append( 1.0 - loss.item() )
            dice = np.mean(loss_valid)
            text = 'Epoch: {}. Validation. Loss: {:4.5f}'.format(epoch, dice)
            stream.set_description(text)
            logging.info(text)

            if args.plot:
                plot_epoch(x, y_true, y_pred)

    return dice

def plot_epoch(x, y_true, y_pred):
    b = x.shape[0]
    x_np = x.detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    # print('pred:', y_pred_np.shape)
    # print('true:',y_true_np.shape)

    for k in range(b):
        # print(np.min(x_np[k,0,:]), np.max(x_np[k,0,:]))
        plt.subplot(b,3,3*k+1)
        plt.imshow(x_np[k,0,:])
        plt.subplot(b,3,3*k+2)
        plt.imshow(y_true_np[k,0,:])
        plt.subplot(b,3,3*k+3)
        plt.imshow(y_pred_np[k,0,:])
    plt.pause(0.0001)
    # plt.show()
    return

def makedirs(args):
    os.makedirs(os.path.dirname(args.weights), exist_ok=True)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training U-Net model for segmentation')
    parser.add_argument('--input', '-i', type=str, default='./data/', 
        help='Input path with image and mask folder')
    parser.add_argument('--image', type=str, default='image', 
        help='Image folder')
    parser.add_argument('--mask', type=str, default='tumor', 
        help='Mask folder')
    parser.add_argument('--num-images', type=int, default=-1,
        help='Number of images to include from data folders. If < 1 include all. Default -1.')
    
    
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
    parser.add_argument('--validation', dest='val', type=float, default=0.2,
        help='Percent of the data that is used as validation (0-1)')
    parser.add_argument('--validation-step', type=int, default=1,
        help='Validation jump step. (default: 1)')

    parser.add_argument('--disable-augment', action='store_true',
                        help='Augment')
    parser.add_argument('--disable-noise', action='store_true',
                        help='Augmentation noise')
    parser.add_argument('--overwrite', '-w', action='store_true',
                        help='Overwrite output model')
    parser.add_argument('--plot', '-p', action='store_true',
                        help='Enable plot mode')
    
    parser.add_argument('--weights', type=str, default='./weights/unet.pt', 
        help='folder to save weights')
    parser.add_argument('--log', type=str, default='./log/training.log', 
        help='folder to save logs')
    
    parser.add_argument('--image-size', type=int, default=256,
        help='target input image size (default: 256)')
    # parser.add_argument('--aug-scale', type=int, default=0.05,
    #     help='scale factor range for augmentation (default: 0.05)')
    # parser.add_argument('--aug-angle', type=int, default=10,
    #     help='rotation angle range in degrees for augmentation (default: 10)')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()

