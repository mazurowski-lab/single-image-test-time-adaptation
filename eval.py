import torch
import torch.nn.functional as F

from dice_loss import dice_coeff
from utils.util import contour_selection

from PIL import Image
import numpy as np
import os
import copy

def rescale(img):
    ma = img.max()
    mi = img.min()
    new = (img-mi)/(ma-mi)
    new = new.clip(0,1) * 255
    return new


def eval_net(net, loader, device, epoch_num="0", draw=False, eval_img=None):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    score_list = []
    tot = 0

    for batch_num, batch in enumerate(loader):
        imgs, true_masks = batch['image'], batch['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)

        dc_list = []
        with torch.no_grad():
            if eval_img is not None:
                mask_pred = net(imgs, eval_img)
            else:
                mask_pred = net(imgs)
            if len(mask_pred[0]) > 1:
                for c in range(len(mask_pred[0])):
                    if c == 0:
                        continue
                    pred = torch.sigmoid(mask_pred[:,c])
                    pred = (pred > 0.5).float()
                    curr_masks = (true_masks == c).float()
                    dc = dice_coeff(pred, curr_masks.squeeze().unsqueeze(0)).item()
                    dc_list.append(dc)
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()

                dc = dice_coeff(pred, true_masks).item()
                dc_list.append(dc)

                #if torch.sum(true_masks) == 0:
                #    print(batch['path'])
                #    print(torch.sum(pred))
                #    print(dc)
            
            if len(dc_list) == 1:
                score_list.append(dc_list[0])
            else:
                score_list.append(dc_list)

            if draw:
                if batch_num in [0,1,2,3]:
                    print(dc)
                    print(batch['path'])
                    tmp = Image.fromarray(true_masks[0].cpu().numpy().squeeze()*255).convert('L')
                    tmp.save('figs/mask_%s.png' % batch_num)
                    tmp = Image.fromarray(imgs[0].cpu().numpy().squeeze()*255).convert('L')
                    tmp.save('figs/ori_%s.png' % batch_num)
                    tmp = Image.fromarray(pred[0].cpu().numpy().squeeze()*255).convert('L')
                    tmp.save('figs/pred_%s.png' % batch_num)

    net.train()
    score_list = np.array(score_list)
    return score_list.mean(), score_list
