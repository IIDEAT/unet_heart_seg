import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from loss.dice_loss import dice_coeff
import loss.lovaszloss as L
import loss.dice_loss_mul as dm
import utils.cal_weight as W

from unet import UNet
one_stage_model = './logs/best_yallow.pth'

def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot, tot_loss, tot_dice = 0, 0, 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)
            weight_net = UNet(n_channels=1, n_classes=1, bilinear=True)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            weight_net.to(device=device)
            weight_net.load_state_dict(torch.load(one_stage_model, map_location=device))
            masks_weight = W.predict_img(weight_net, imgs, device)

            imgs = masks_weight * imgs
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)
                mask_pred[:,0,:,:][masks_weight[:,0,:,:]==0] += 100
            if net.n_classes > 1:
                # print(np.unique(F.softmax(mask_pred, dim=1).cpu().numpy()))
                # print(np.unique(true_masks.cpu().numpy()))
                dice = dm.DiceLoss()
                tot_dice += dice(mask_pred, true_masks)
                mask_pred = F.softmax(mask_pred, dim=1)
                tot_loss += L.lovasz_softmax(mask_pred, true_masks)
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_masks).item()
            pbar.update()

    net.train()
    return tot_loss / n_val, 1 - tot_dice / n_val
