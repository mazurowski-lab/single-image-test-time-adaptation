import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Function


# Ours, need reparametric trick!
def kl_gaussian_loss(mu, logvar):
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_loss


# faster convolutions, but more memory
# cudnn.benchmark = True
def get_batch(dataset, batch_size):
    x_list = []
    spinal_mask_list = []
    gm_mask_list = []
    for _ in range(batch_size):
        batch = dataset[0]
        x = batch['image']
        mask = batch['mask']
        if 'mask_gm' in batch.keys():
            gm_mask = batch['mask_gm']
            gm_mask_list.append(gm_mask)
        x_list.append(x)
        spinal_mask_list.append(mask)

    if len(gm_mask_list) == 0:
        return x_list, spinal_mask_list
    else:
        return x_list, spinal_mask_list, gm_mask_list

'''
def get_batch(dataset, batch_size):
        x_list = []
        spinal_mask_list = []
        gm_mask_list = []
        for _ in range(batch_size):
            x, spinal_cord_mask, gm_mask = dataset[0]
            x_list.append(x)
            spinal_mask_list.append(spinal_cord_mask)
            gm_mask_list.append(gm_mask)
        return x_list, spinal_mask_list, gm_mask_list
    # return torch.stack(x_list, dim=0).cuda(), torch.stack(spinal_mask_list, dim=0).cuda(), torch.stack(gm_mask_list,dim=0).cuda()
'''

def csa_loss(x, y, class_eq):
    margin = 1
    dist = F.pairwise_distance(x, y)
    loss = class_eq * dist.pow(2)
    loss += (1 - class_eq) * (margin - dist).clamp(min=0).pow(2)
    return loss.mean()


class LowRank(Function):
    @staticmethod
    def forward(ctx, x):
        U, S, V = torch.svd(x)
        ctx.save_for_backward(x, U, V)
        return torch.sum(S)

    @staticmethod
    def backward(ctx, grad_output):
        data = ctx.saved_tensors
        grad = torch.mm(data[1], data[2].t())
        return grad_output * grad

def get_multi_batch(dataset_list, batch_size):
    x_list = []
    spinal_mask_list = []
    gm_mask_list = []
    for dataset in dataset_list:
        out = get_batch(dataset, batch_size)
        x = out[0]
        spinal_cord_mask = out[1]
        if len(out) > 2:
            gm_mask = out[2]
            gm_mask_list.extend(gm_mask)
        x_list.extend(x)
        spinal_mask_list.extend(spinal_cord_mask)
    if len(gm_mask_list) == 0:
        return torch.stack(x_list, dim=0).cuda(), torch.stack(spinal_mask_list, dim=0).cuda()
    else:
        return torch.stack(x_list, dim=0).cuda(), torch.stack(spinal_mask_list, dim=0).cuda(), torch.stack(gm_mask_list, dim=0).cuda()

def contour_selection(mask_pred, center):
    mask_img = (mask_pred*255).cpu().permute(1,2,0).numpy() 
    mask_img = np.uint8(mask_img)
    ret, thresh = cv2.threshold(mask_img, 50, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        mask_loc = np.zeros_like(mask_img)
        mask_loc[int(center[0]), int(center[1])] = 1 
        for c in contours:
            tmp = np.zeros_like(mask_img)
            cv2.drawContours(tmp, c, -1, (255,255,255), -1) 
            tmp = (tmp != 0)
            cover = np.sum(mask_loc * tmp)
            if cover:
                print('only center contour select')
                tmp = torch.tensor(tmp).permute(2,0,1).cuda()
                mask_pred = mask_pred * tmp 
                return mask_pred
