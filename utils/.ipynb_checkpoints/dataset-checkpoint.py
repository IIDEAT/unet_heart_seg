from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
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
    def augment(cls, img, mask):
        # print(img.dtype)
        # print(mask.dtype)
        mask = mask.astype('uint8')
        # ROTATED
        img_ag = []
        for i in range(5):
          img_after = img[i,:,:]
          if random.random() < 0.5:
              theta = random.uniform(-10,10)
              M = cv2.getRotationMatrix2D((img_after.shape[1] / 2, img_after.shape[0] / 2), theta, 1)
              img_after = cv2.warpAffine(img_after, M, (img_after.shape[1], img_after.shape[0]))
              if i==0:
                mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
        # SHIFTED
          if random.random() < 0.5:
              tx = random.randint(-20,20)
              ty = random.randint(-20,20)
              affine_arr = np.float32([[1,0,tx],[0,1,ty]])
              img_after = cv2.warpAffine(img_after,affine_arr,(img_after.shape[0],img_after.shape[1]))
              if i==0:
                mask = cv2.warpAffine(mask,affine_arr,(mask.shape[0],mask.shape[1]))
            # print(img.shape)
            # print(mask.shape)
          img_ag.append(img_after)

        img = np.stack(img_ag,axis=0)
        return img, mask

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        # print(img_trans)
        # print(type(img_trans))
        img_trans = cls.augment(img_trans)
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    @classmethod
    def preprocess_mask(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        # print(img_trans)
        # print(type(img_trans))

        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = np.load(mask_file[0],allow_pickle=True)
        img = np.load(img_file[0])
        # img_ag = []
        # for i in range(5):
        #   img_after = img[i,:,:]
        #   mean = np.mean(img_after)
        #   var = np.mean(np.square(img_after-mean))
        #   img_after = (img_after - mean)/np.sqrt(var)
        #   img_after = np.maximum(img_after, 0)
        #   img_ag.append(img_after)
        # img = np.stack(img_ag, axis=0)
        # img, mask = self.augment(img,mask)
        # print(mask.shape)
        # print(img.shape)
        # img = np.transpose(img, (2, 0, 1))
        img = img/255
        # mask = mask / 255
        # mask = mask -1 
        # print(np.unique(mask))
        # mask = np.expand_dims(mask, axis=0)
        # print(img.shape)
        # mask = mask.transpose(())
        # assert img.size == mask.size, \
        #     f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        # img = self.preprocess(img, self.scale)
        # mask = self.preprocess_mask(mask, self.scale)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
