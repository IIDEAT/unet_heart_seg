import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys
# 黄1 红2 绿3
color = [[255, 242, 0], [237, 28, 36], [34, 177, 76]]
for dir in os.listdir('./ori_label/'):
    if not os.path.exists('./npy_label/' + dir):
        os.mkdir('./npy_label/' + dir)
    for file in os.listdir('./ori_label/' + dir):
        # print(dir, file)
        label = np.zeros((512, 512, 3))
        img = cv2.imread('./ori_label/' + dir + '/' + file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                for idx in range(1, 4):
                    if (img[x, y] == color[idx - 1]).all() == True:
                        label[x, y, idx - 1] = 1
        if len(np.unique(label)) > 1:
            for idx in range(1, 4):
                if 1 in label[:, :, idx - 1]:
                    label_temp = label[:, :, idx - 1].copy()
                    label_temp = label_temp.astype('uint8')
                    contours, hierarchy = cv2.findContours(label_temp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                    hierarchy = hierarchy[0]
                    for i, c in enumerate(contours):
                        if hierarchy[i][2] < 0 and hierarchy[i][3] < 0:
                            print('no', dir ,file, idx)
                    #         sys.exit(0)
                    # cv2.drawContours(label_temp, [contours[0]], -1, 1, thickness=-1)
                    # label[:, :, idx - 1] = label_temp
                    # plt.imshow(label[:, :, idx-1])
                    # plt.show()
            np.save('./npy_label/' + dir + '/' + file[:-4] + '.npy', label)

