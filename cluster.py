import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.cluster import *
from sklearn.mixture import GaussianMixture as GMM
from PIL import Image

image_path = './data/imgs/val2/'
mask_path = './data/predict/'

def cluster(img, mask, color):
    # 237 34
    # red green
    clahe1 = cv2.createCLAHE(clipLimit=1000.0, tileGridSize=(3, 3))
    clahe2 = cv2.createCLAHE(clipLimit=1000.0, tileGridSize=(1, 1))
    ori_img = img.copy()
    ori_img = ori_img / ori_img.max() * 255
    ori_img = ori_img.astype('uint8')
    ori_img = clahe1.apply(img)
    if len(img[mask[:, :, 0] == 237]) == 0 or len(img[mask[:, :, 0] == 34]) == 0:
        return
    img[(mask[:, :, 0] != 237) & (mask[:, :, 0] != 34)] = 0
    red = clahe2.apply(img[mask[:, :, 0] == 237])
    img[mask[:, :, 0] == 237] = red.reshape(-1)
    green = clahe2.apply(img[mask[:, :, 0] == 34])
    img[mask[:, :, 0] == 34] = green.reshape(-1)
    clu = img.copy()
    num = 3

    pred_red = GMM(n_components=num).fit_predict(img[mask[:, :, 0] == 237].reshape(-1, 1))
    pred_green = GMM(n_components=num).fit_predict(img[mask[:, :, 0] == 34].reshape(-1, 1))
    l = []
    g = []
    for i in range(3):
        l.append(np.mean(img[mask[:, :, 0] == 237][pred_red == i]))
        g.append(np.mean(img[mask[:, :, 0] == 34][pred_green == i]))
    l = np.argsort(-np.array(l))
    g = np.argsort(-np.array(g))
    pred_after_red = np.zeros_like(pred_red)
    pred_after_green = np.zeros_like(pred_green)
    for i in range(3):
        pred_after_red[pred_red == i] = l[i]
        pred_after_green[pred_green == i] = g[i]
    clu[mask[:, :, 0] == 237] = pred_after_red
    clu[mask[:, :, 0] == 34] = pred_after_green
    plt.imshow(clu, cmap='tab10', alpha=1)
    plt.colorbar()
    plt.imshow(ori_img, cmap='bone', alpha=0.5)
    plt.imshow(img, cmap='bone', alpha=0.5)
    plt.show()


def seed_growth(img, mask):
    ori_img = img.copy()
    img[(mask[:, :, 0] != 237) & (mask[:, :, 0] != 34)] = 0
    red = (mask[:, :, 0] != 237)
    green = (mask[:, :, 0] != 34)
    img[green] = 0
    contours, h = cv2.findContours(img.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seeds = [[contours[0][0][0][1], contours[0][0][0][0]], [contours[0][1][0][1], contours[0][1][0][0]]]
    mask = np.zeros_like(img)
    mask[img != 0] = 2
    mask[seeds[0][0], seeds[0][1]] = 2
    mask[seeds[1][0], seeds[1][1]] = 2
    connects = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), \
                (0, 1), (-1, 1), (-1, 0)]
    # connects = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    th = 10
    print('th', th)
    clahe2 = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(3, 3))
    temp = clahe2.apply(img[img != 0])
    img[img != 0] = temp.reshape(-1)
    while len(seeds) != 0:
        pt = seeds.pop(0)
        for i in range(len(connects)):
            tmpy = pt[0] + connects[i][0]
            tmpx = pt[1] + connects[i][1]

            if tmpx < 0 or tmpy < 0 or tmpx >= img.shape[1] - 1 or tmpy >= img.shape[0] - 1:
                continue
            if abs(int(img[pt[0], pt[1]]) - int(img[tmpy, tmpx])) <= th and mask[tmpy, tmpx] == 2:
                mask[tmpy, tmpx] = 1
                seeds.append([tmpy, tmpx])
    yxy, yxx = max(np.where(mask != 0)[0]), max(np.where(mask != 0)[1])
    ysy, ysx = min(np.where(mask != 0)[0]), min(np.where(mask != 0)[1])
    ax1 = plt.subplot(1, 2, 1, frameon=False)
    plt.imshow(mask[ysy - 5:yxy + 5, ysx - 5:yxx + 5], cmap='Set1')
    plt.colorbar()
    # plt.show()
    ax2 = plt.subplot(1, 2, 2, frameon=False)
    plt.imshow(img[ysy - 5:yxy + 5, ysx - 5:yxx + 5], cmap='bone')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    flag = 1
    for file in os.listdir(image_path):
        print(file)
        if flag:
            # flag = 0
            img = np.load(image_path + file)
            img = img[0]
            # plt.imshow(img, cmap='bone')
            mask = np.load(mask_path + file[:-4] + 'pred.npy')
            mask = mask[0]
            mask = np.transpose(mask, [1, 2, 0])
            seed_growth(img, mask)