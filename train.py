import os
import argparse
import json
import logging
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import SimpleDataset
from loss import DiceLoss
from transform import transforms
from unet import UNet
from utils import log_images, dsc

import matplotlib.pyplot as plt

def main():
    args = parse_arguments()
    makedirs(args)
    snapshotargs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    # Dataset
    folder_image = os.path.join(args.input, 'images')
    folder_mask = os.path.join(args.input, 'masks')
    dataset = SimpleDataset(folder_image, folder_mask, args.image_size)
    
    # Split into train / validation partitions
    n_val = int(len(dataset) * args.val)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # Create data loaders
    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=False)
    loader_train = DataLoader(train_set, shuffle=True, **loader_args)
    loader_valid = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    loaders = {"train": loader_train, "valid": loader_valid}

    file_log = os.path.join(args.logs,'training.log')
    logging.basicConfig(filename=file_log, level=logging.INFO, 
        filemode='w', encoding='utf-8', format='%(levelname)s: %(message)s')
    logging.info(f'''Starting training:
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Learning rate:   {args.learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}''')

    unet = UNet(in_channels=SimpleDataset.in_channels, out_channels=SimpleDataset.out_channels)
    unet.to(device)

    dsc_loss = DiceLoss()
    best_validation_dsc = 0.0

    optimizer = optim.Adam(unet.parameters(), lr=args.learning_rate)

    loss_train = []
    loss_valid = []

    step = 0
    plt.figure()

    for epoch in tqdm(range(args.epochs), total=args.epochs):
        for phase in ["train", "valid"]:
            if phase == "train":
                unet.train()
            else:
                unet.eval()

            validation_pred = []
            validation_true = []

            for i, data in enumerate(loaders[phase]):
                if phase == "train":
                    step += 1

                x, y_true = data
                x, y_true = x.to(device), y_true.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = unet(x)
                    # print(y_pred.shape)
                    # print(y_true.shape)
                    loss = dsc_loss(y_pred, y_true)

                    if phase == "valid":
                        loss_valid.append(loss.item())
                        y_pred_np = y_pred.detach().cpu().numpy()
                        validation_pred.extend( [y_pred_np[s] for s in range(y_pred_np.shape[0])] )
                        
                        y_true_np = y_true.detach().cpu().numpy()
                        validation_true.extend([y_true_np[s] for s in range(y_true_np.shape[0])] )

                        # print(y_true_np.shape)
                        bb = args.batch_size
                        for i in range(bb):
                            plt.subplot(bb,2,bb*i+1)
                            plt.imshow(y_true_np[i,0,:])
                            plt.subplot(bb,2,bb*i+2)
                            plt.imshow(y_pred_np[i,0,:])
                        plt.pause(0.0001)
                        # if (epoch % args.vis_freq == 0) or (epoch == args.epochs - 1):
                        #     if i * args.batch_size < args.vis_images:
                        #         tag = "image/{}".format(i)
                        #         num_images = args.vis_images - i * args.batch_size
                                # logger.image_list_summary(tag, 
                                #     log_images(x, y_true, y_pred)[:num_images], 
                                #     step)

                    if phase == "train":
                        loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()

                if phase == "train" and (step + 1) % 10 == 0:
                    # log_loss_summary(logger, loss_train, step)
                    logging.info('Step: {}. Train Loss: {}.'.format(step,loss_train))
                    loss_train = []

            if phase == "valid":
                mean_dsc = np.mean(loss_valid)*-1.0
                print(mean_dsc)
                # mean_dsc = np.mean( dsc_validation(validation_pred, validation_true) )
                # mean_dsc = np.mean( dsc_validation(validation_pred, validation_true) )
                # log_loss_summary(logger, loss_valid, step, prefix="val_")
                # logger.scalar_summary("val_dsc", mean_dsc, step)
                logging.info('Step: {}. Validation Loss: {}.'.format(step,mean_dsc))
                if mean_dsc > best_validation_dsc:
                    best_validation_dsc = mean_dsc
                    torch.save(unet.state_dict(), os.path.join(args.weights, "unet.pt"))
                loss_valid = []

    print("Best validation mean DSC: {:4f}".format(best_validation_dsc))


# def data_loaders(args):
#     dataset_train, dataset_valid = datasets(args)

#     def worker_init(worker_id):
#         np.random.seed(42 + worker_id)

#     loader_train = DataLoader(
#         dataset_train,
#         batch_size=args.batch_size,
#         shuffle=True,
#         drop_last=True,
#         num_workers=args.workers,
#         worker_init_fn=worker_init,
#     )
#     loader_valid = DataLoader(
#         dataset_valid,
#         batch_size=args.batch_size,
#         drop_last=False,
#         num_workers=args.workers,
#         worker_init_fn=worker_init,
#     )

#     return loader_train, loader_valid


# def datasets(args):
#     train = Dataset(
#         images_dir=os.path.join(args.input,'images'),
#         subset="train",
#         image_size=args.image_size,
#         transform=transforms(scale=args.aug_scale, angle=args.aug_angle, flip_prob=0.5),
#     )
#     valid = Dataset(
#         images_dir=os.path.join(args.input,'images'),
#         subset="validation",
#         image_size=args.image_size,
#         random_sampling=False,
#     )
#     return train, valid


# def dsc_validation(validation_pred, validation_true):
#     dsc_list = []
#     for p in range(len(validation_pred)):
#         y_pred = np.array(validation_pred[p])
#         y_true = np.array(validation_true[p])
#         dsc_list.append(dsc(y_pred, y_true))
#     return dsc_list

# def dsc_validation(validation_pred, validation_true):
#     dsc_loss = DiceLoss()
#     dsc_list = []
#     for p in range(len(validation_pred)):
#         y_pred = np.array(validation_pred[p])
#         y_true = np.array(validation_true[p])
#         dsc_list.append(dsc_loss(y_pred, y_true))
#     return dsc_list


# def log_loss_summary(logger, loss, step, prefix=""):
#     logger.scalar_summary(prefix + "loss", np.mean(loss), step)

def makedirs(args):
    os.makedirs(args.weights, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)

def snapshotargs(args):
    args_file = os.path.join(args.logs, "args.json")
    with open(args_file, "w") as fp:
        json.dump(vars(args), fp)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Training U-Net model for segmentation')
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

