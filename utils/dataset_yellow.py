from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from torchvision import transforms
from PIL import Image
import random
import cv2
import matplotlib.pyplot as plt


np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed(20)
torch.cuda.manual_seed_all(20)
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
    def preprocess(cls, img, mask, scale):
        # print(pil_mask.size)
        img = img.astype('uint16')
        mask = mask.astype('uint8')
        clahe = cv2.createCLAHE(100, (1, 1))
        img_after = clahe.apply(img)
        img_after = (img_after - np.mean(img_after)) / np.std(img_after)
        if np.random.random() < 0.5:
            theta = random.uniform(-10,10)
            M = cv2.getRotationMatrix2D((img_after.shape[1] / 2, img_after.shape[0] / 2), theta, 1)
            img_after = cv2.warpAffine(img_after, M, (img_after.shape[1], img_after.shape[0]))
            mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))

        if np.random.random() < 0.5:
            tx = random.randint(-10, 10)
            ty = random.randint(-10, 10)
            affine_arr = np.float32([[1, 0, tx], [0, 1, ty]])
            img_after = cv2.warpAffine(img_after, affine_arr, (img_after.shape[0], img_after.shape[1]))
            mask = cv2.warpAffine(mask, affine_arr, (mask.shape[0], mask.shape[1]))

        if len(img_after.shape) == 2:
            img_after = np.expand_dims(img_after, axis=0)
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=0)
        return img_after, mask



    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')
        # print(mask_file)
        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = np.load(mask_file[0])
        img = np.load(img_file[0])
        img = img[0, :, :]
        img, mask = self.preprocess(img, mask, self.scale)
        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
