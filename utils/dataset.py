from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from unet import UNet
import utils.cal_weight as W
import random
import cv2
np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed(20)
torch.cuda.manual_seed_all(20)
random.seed(20)
class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix='mask'):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, imgs, mask):
        # print(pil_mask.size)
        one_stage_model = './logs/CP_epoch43yellow.pth'

        imgs = imgs.astype('uint16')
        mask = mask.astype('uint8')
        clahe = cv2.createCLAHE(100, (1, 1))
        imgs_after = np.zeros_like(imgs).astype('float32')
        for i in range(imgs.shape[0]):
            img = imgs[i, :, :]
            img = clahe.apply(img)
            img = (img - np.mean(img)) / np.std(img)
            imgs_after[i, :, :] = img
            if np.random.random() < 0.5:
                theta = random.uniform(-10, 10)
                M = cv2.getRotationMatrix2D((imgs_after[i, : ,:].shape[1] / 2, imgs_after[i, : ,:].shape[0] / 2), theta, 1)
                imgs_after[i, : ,:] = cv2.warpAffine(imgs_after[i, :, :], M, (imgs_after[i, :, :].shape[1], imgs_after[i, :, :].shape[0]))
                if i == 0:
                    mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
            if np.random.random() < 0.5:
                tx = random.randint(-10, 10)
                ty = random.randint(-10, 10)
                affine_arr = np.float32([[1, 0, tx], [0, 1, ty]])
                imgs_after[i, :, :] = cv2.warpAffine(imgs_after[i, :, :] , affine_arr,
                                               (imgs_after[i, : ,:] .shape[0], imgs_after[i, :, :] .shape[1]))
                if i == 0:
                    mask = cv2.warpAffine(mask, affine_arr, (mask.shape[0], mask.shape[1]))
        return imgs_after, mask


    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = np.load(mask_file[0])
        imgs = np.load(img_file[0])
        imgs, mask = self.preprocess(imgs, mask)
        return {
            'image': torch.from_numpy(imgs).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
