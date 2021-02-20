import cv2
import os
from PIL import Image
import numpy as np

path = './all_mask/'


def mask_vis(mask):
    # NONE, YELLOW, RED, GREEN, PURPLE, BLUE
    palette = [[0, 0, 0], [237, 28, 36],
               [34, 177, 76], [163, 73, 164], [63, 72, 204]]
    n = mask.shape[0]
    img_batch = []
    for i in range(n):
        mask_temp = mask[i, :, :]
        mask_new = Image.new("RGB", (272, 272), (0, 0, 0))

        r, g, b = mask_new.split()

        r = np.array(r)
        g = np.array(g)
        b = np.array(b)

        for i in range(1, 5):
            color = np.where(mask_temp == i)
            r[color] = palette[i][0]
            g[color] = palette[i][1]
            b[color] = palette[i][2]

        r = Image.fromarray(r)
        g = Image.fromarray(g)
        b = Image.fromarray(b)

        mask1 = Image.merge("RGB", [r, g, b])
        mask1 = np.array(mask1)
        img_batch.append(mask1)
    final = np.array(img_batch)
    final = final.transpose((0, 3, 1, 2))
    # print(final.shape)
    return final
    # mask1.save('./temp_temp/' + file)


def onehot2mask(onehot):
    # print(onehot.shape)
    onehot = onehot + 0
    onehot_temp = onehot
    n = onehot.shape[0]
    print(onehot.shape)
    img_batch = []
    for i in range(n):
      onehot_temp = onehot[i, :, :, :]
      # print(onehot_temp.shape[0])
      palette = [[0, 0, 0], [237, 28, 36],
                [34, 177, 76], [163, 73, 164], [63, 72, 204]]
      x = np.argmax(onehot_temp, axis=0)
      colour_codes = np.array(palette)
      x = np.uint8(colour_codes[x.astype(np.uint8)])
      # print(x.shape)
      img_batch.append(x)
    final = np.array(img_batch)
    print(final.shape)
    final = final.transpose((0, 3, 1, 2))
    print(np.unique(final))
    return final

def make_mask(mask, file):
    # NONE, YELLOW, RED, GREEN, PURPLE, BLUE
    palette = [[0, 0, 0], [255, 242, 0], [237, 28, 36],
               [34, 177, 76], [163, 73, 164], [63, 72, 204]]

    path = './all_mask/'
    n = 2
    for _ in ['r_c/', 'g_c/', 'p_c/', 'b_c/']:
        # print(_)
        mask_color = Image.open(path + _ + _[0:2] + file)
        mask_color = mask_color.convert('1')
        mask_color = np.array(mask_color) + 0
        mask_color_idx = np.where(mask_color == 1)
        mask[mask_color_idx] = n
        n += 1
        # mask_color = mask_color / 255
        # print(mask_color.shape)
        # print(mask.shape)
    # print(np.unique(mask))
    return mask


if __name__ == '__main__':
    for file in os.listdir(path):
        # print(file)
        if os.path.isfile(path + file):
            print(file)
            mask = Image.open(path + file)
            mask = np.array(mask)
            mask = mask / 255
            # print(mask.shape)
            mask = make_mask(mask, file)
            np.save('./mask_np_all/' + file[0:-4], mask)
            # onehot = np.load('./mask_np_all/' + file[0:-4] + '.npy')
            # print(onehot.shape)
            mask_vis(mask, file)