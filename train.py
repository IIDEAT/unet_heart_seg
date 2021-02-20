import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from loss.combo import comboloss
import loss.dice_loss_mul as dm
from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

import torch.nn.functional as F
import utils.onehot_mask as onehot_mask

import loss.lovaszloss as L
import cv2
import utils.cal_weight as W
dir_img_train = './data/imgs/train/'
dir_mask_train = './data/masks/train/'
dir_img_val = './data/imgs/val/'
dir_mask_val = './data/masks/val/'
dir_checkpoint = './logs/'
one_stage_model = './logs/best_yallow.pth'

np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed(20)
torch.cuda.manual_seed_all(20)
def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):

    train = BasicDataset(dir_img_train, dir_mask_train, img_scale)
    val = BasicDataset(dir_img_val, dir_mask_val, img_scale)
    n_val = len(val)
    n_train = len(train)
    # train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    print("#################")
    val_loader = DataLoader(val, batch_size=batch_size,shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

#     optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.AdamW(net.parameters(), lr=lr)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)

    criterion = nn.CrossEntropyLoss(reduction='none')
    flag=0
    max_score = 0
    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='bmp') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                # print(np.unique(imgs.cpu().numpy()))
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
                weight_net = UNet(n_channels=1, n_classes=1, bilinear=True)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                weight_net.to(device=device)
                weight_net.load_state_dict(torch.load(one_stage_model, map_location=device))
                masks_weight = W.predict_img(weight_net, imgs, device)
                imgs = masks_weight * imgs
                # for i in range(5):
                #     clahe = cv2.createCLAHE(100, (1, 1))
                #     imgs[i, :, :]
                masks_pred = net(imgs)
                masks_pred[:,0,:,:][masks_weight[:,0,:,:]==0] += 100
                # masks_pred_temp = F.softmax(masks_pred_temp, dim=1)
                # loss = criterion(masks_pred, true_masks)
                # print(loss)
                # print(np.unique(masks_weight.cpu().numpy()))
                # loss = masks_weight * loss
                # print(loss.shape)
                # print(loss)
                loss = comboloss(masks_pred, true_masks)
                # loss = L.lovasz_softmax(F.softmax(masks_pred, dim=1), true_masks, ignore=0)
#                 dice = dm.DiceLoss()
#                 loss = dice(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
        for tag, value in net.named_parameters():
            tag = tag.replace('.', '/')
            writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
            writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
        val_score, val_score_dice = eval_net(net, val_loader, device)
#                     scheduler.step(val_score)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

        if net.n_classes > 1:
            logging.info('Validation Lovasz Softmax: {}'.format(val_score))
            writer.add_scalar('Loss/test', val_score, global_step)
            logging.info('Validation Dice Coeff: {}'.format(val_score_dice))
            writer.add_scalar('Dice/test', val_score_dice, global_step)
        else:
            logging.info('Validation Dice Coeff: {}'.format(val_score))
            writer.add_scalar('Dice/test', val_score_dice, global_step)
            # np.stack((img,)*3, axis=-1)
        writer.add_images('images', np.stack((imgs[:,0,:,:].cpu().numpy() ,)*3,axis=1), global_step)
        if net.n_classes == 1:
            writer.add_images('masks/true', true_masks, global_step)
            writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
        else:
            # print(F.softmax(masks_pred, dim=1) > 0.5)
            # print((F.softmax(masks_pred, dim=1)> 0.5).shape)
            # print(np.unique((F.softmax(masks_pred, dim=1) ).cpu().numpy()))
            writer.add_images('masks/true', onehot_mask.mask_vis(true_masks.cpu().numpy()), global_step)
            writer.add_images('masks/pred', onehot_mask.onehot2mask((F.softmax(masks_pred, dim=1) > 0.5).cpu().numpy()), global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'latest.pth')
            print(val_score_dice, max_score)
            if val_score_dice > max_score:
                max_score = val_score_dice
                torch.save(net.state_dict(),
                           dir_checkpoint + f'best.pth')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=500,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=16,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0015,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=5, n_classes=5, bilinear=True)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
