import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import re
path = './data/predict/'
ori_image_path = './data/imgs/val2/'
# for file in os.listdir(path):
#     img = np.load(path + file)
#     img = img[0, :, :, :]
#     img = np.transpose(img, [1, 2, 0])
#     plt.imshow(img)
#     plt.show()
    # if 'weight' in file:
    #     img = img[:, :, 0]
    #     img *= 255
    #     bmp = Image.fromarray(img)
    #     bmp = bmp.convert('RGB')
    #     bmp.save('./ymz/data/predict/picture/ori/' + file[:-4] + '.bmp')
    # else:
    #     bmp = Image.fromarray(img)
    #     bmp.save('./ymz/data/predict/picture/ori/' + file[:-4] + '.bmp')

for file in os.listdir(ori_image_path):
    print(file)
    img = np.load(ori_image_path + file)
    img = img[0, :, :]
    for mask in os.listdir(path):
        if file[:-4] == mask[0:len(file[:-4])] and 'pred' in mask:
            print(mask)
            mask_img = np.load(path + mask)
            mask_img = mask_img[0]
            mask_img = np.transpose(mask_img, [1, 2, 0])
            plt.imshow(mask_img, alpha=0.5)
            plt.imshow(img, alpha=0.7, cmap='bone')
            plt.savefig(path + 'picture/patch/' + file[:-4] + 'pred.jpg')
            plt.close()
        if file[:-4] == mask[0:len(file[:-4])] and 'true' in mask:
            print(mask)
            mask_img = np.load(path + mask)
            mask_img = mask_img[0]
            mask_img = np.transpose(mask_img, [1, 2, 0])
            plt.imshow(mask_img, alpha=0.7)
            plt.imshow(img, alpha=0.7, cmap='bone')
            plt.savefig(path + 'picture/patch/' + file[:-4] + 'true.jpg')
            plt.close()
