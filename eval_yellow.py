import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from loss.dice_loss import dice_coeff
import loss.lovaszloss as L
import loss.dice_loss_mul as dm
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
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)
            # print(mask_pred)
            if net.n_classes > 1:
                # print(np.unique(F.softmax(mask_pred, dim=1).cpu().numpy()))
                # print(np.unique(true_masks.cpu().numpy()))
                dice = dm.DiceLoss()
                tot_dice += dice(mask_pred, true_masks)
                mask_pred = F.softmax(mask_pred, dim=1)
                tot_loss += L.lovasz_softmax(mask_pred, true_masks)
            else:
                pred = torch.sigmoid(mask_pred)
                # print(pred)
                pred = (pred > 0.4).float()
                tot += dice_coeff(pred, true_masks).item()
            pbar.update()

    net.train()
    return tot / n_val
