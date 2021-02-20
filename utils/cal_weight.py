import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset


def predict_img(net,
                imgs,
                device,
                        ):
    net.eval()
    weights = []
    for i in range(5):
      img = imgs[:, i, :, :]
#       print(img.shape)
      img = img.unsqueeze(1)
#       print(img.shape)
      with torch.no_grad():
        output = net(img)
        probs = torch.sigmoid(output)
        probs = probs.squeeze(0)
        mask = probs.squeeze(1).cpu().numpy() > 0.5
        mask = mask + 0
        mask[mask==0] = 0
        mask[mask==1] = 10
        mask = mask/10
        weights.append(mask)
    weights = np.stack(weights, axis=1)
    weights = torch.from_numpy(weights).type(torch.FloatTensor)
    weights = weights.to(device=device, dtype=torch.float32)

    return weights