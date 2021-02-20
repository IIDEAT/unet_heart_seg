import argparse
import logging
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

import torch.nn.functional as F
import utils.onehot_mask as onehot_mask
from loss.dice_loss import dice_coeff
import loss.lovaszloss as L
import loss.dice_loss_mul as dm

import utils.cal_weight as W

np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed(20)
torch.cuda.manual_seed_all(20)

test_dir = './data/imgs/val2/'
test_dir2 = './data/predict/'
one_stage_model = './logs/best_yallow.pth'
two_stage_model = './logs/best.pth'
one_net = UNet(n_channels=1, n_classes=1, bilinear=True)
two_net = UNet(n_channels=5, n_classes=5, bilinear=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

one_net.to(device=device)
two_net.to(device=device)

one_net.load_state_dict(torch.load(one_stage_model, map_location=device))
two_net.load_state_dict(torch.load(two_stage_model, map_location=device))

test_list = os.listdir(test_dir)
test_list.sort(key = lambda x: int(x[:-4]))
sta=0
for img_name in test_list:
    if len(img_name)<=8:

        imgs = np.load(test_dir + img_name)
        imgs = imgs.astype('uint16')
        clahe = cv2.createCLAHE(100, (1, 1))
        imgs_after = np.zeros_like(imgs).astype('float32')
        for i in range(imgs.shape[0]):
            img = imgs[i, :, :]
            img = clahe.apply(img)
            img = (img - np.mean(img)) / np.std(img)
            imgs_after[i, :, :] = img
        img = imgs_after
        true_mask = np.load('./data/masks/val2/' + img_name[:-4]+'mask.npy')
        true_mask = torch.from_numpy(true_mask).type(torch.FloatTensor)
        img = torch.from_numpy(img).type(torch.FloatTensor)
        true_mask = true_mask.to(device=device, dtype=torch.long)
        img = img.to(device=device, dtype=torch.float32)
        img = img.unsqueeze(0)
    #     print(img.shape)
        weights = W.predict_img(one_net, img, device)
        np.save(test_dir2+img_name[:-4]+'weight.npy', weights.cpu().numpy())
    #     print(weights.shape)
    #     print(img.shape)
        img = weights * img
        mask_pred = two_net(img)
        mask_pred[:,0,:,:][weights[:,0,:,:]==0] += 100
        true_mask = true_mask.unsqueeze(0)
        dice = dm.DiceLoss()
        tot_dice = dice(mask_pred, true_mask)
        print(img_name, 1-tot_dice)
        sta += 1 - tot_dice
        true_mask = onehot_mask.mask_vis(true_mask.cpu().numpy())
        np.save(test_dir2+img_name[:-4]+'true.npy', true_mask)
        mask_pred = onehot_mask.onehot2mask((F.softmax(mask_pred, dim=1) > 0.5).cpu().numpy())
        np.save(test_dir2+img_name[:-4]+'pred.npy', mask_pred)
print(sta/len(test_list))