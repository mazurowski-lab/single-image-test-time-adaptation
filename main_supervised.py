import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import json
import logging
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet, UNetAdvance, UNetVGG
from monai.losses import DiceCELoss, DiceLoss

from utils.dataset import BasicDataset, GMDataset, RetinaDataset
from torch.utils.data import DataLoader, random_split

def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=1.0,
              encoder=None,
              iter_idx=-1, 
              train_class=0):

    if args.dataset == 'gmsc':
        dir_img = '../dataset/gmsc_dataset/images_noblank/'
        dir_mask = '../dataset/gmsc_dataset/masks/'
        
        all_phase = ['site1', 'site2', 'site3', 'site4']
        train_phase = 'site%s' % train_class
        all_phase.remove(train_phase)
        prefix = 'base'

        train = GMDataset(dir_img, dir_mask, sources=[train_phase], val=False, phase='train')
        vals = [GMDataset(dir_img, dir_mask, sources=[train_phase], val=True)]
        vals += [GMDataset(dir_img, dir_mask, sources=[s], full=True) for s in all_phase]

    n_train = len(train)
    n_val = len(vals[0])

    print('*****')
    print(dir_img)
    print('*****')
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loaders = []
    for val in vals:
        val_loaders.append(DataLoader(val, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True))

    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size} 
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')
    
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    criterion = DiceLoss(to_onehot_y=True, sigmoid=True)

    best_val_score = -1
    best_val_list = []
    patient = 0
    
    for epoch in range(epochs):
        net.train()

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for count, batch in enumerate(train_loader):
                imgs = batch['image']
                true_masks = batch['mask']

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if args.out_class == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)

                if args.out_class == 1:
                    pos_weight = torch.tensor(1.) / torch.mean(true_masks.detach()) * 2
                    if torch.isinf(pos_weight) or torch.isnan(pos_weight):
                        pos_weight = torch.tensor(1.).cuda()

                    seg_loss = F.binary_cross_entropy_with_logits(masks_pred, true_masks)
                    dc_loss = criterion(masks_pred, true_masks)
                    loss = dc_loss + seg_loss
                else:
                    loss = criterion(masks_pred, true_masks)
                
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                
            val_list = []
            val_list_full = []
            for i, val_loader in enumerate(val_loaders):
                val_score, score_list = eval_net(net, val_loader, device, str(epoch))
                val_list.append(val_score)
                val_list_full.append(score_list)
                print(' ' + str(i) + ' val score : ' + str(val_score))
            if val_list[0] > best_val_score:
                best_val_score = val_list[0]
                best_val_list = val_list
                best_list_full = val_list_full
                torch.save(net.state_dict(), '../checkpoints/%s_%s_%s_%s_final.pth' % (args.dataset, prefix, train_class, iter_idx))
                patient = 0
            else:
                patient += 1
            print(' best score: ' + str(best_val_list))
            
            if patient == 20:
                break

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=10,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1.0,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--fixed', action='store_true',
                        help='Whether init conv is fixed')
    parser.add_argument('--dataset', default='gmsc', type=str)
    parser.add_argument('--out_class', default=1, type=int)
    parser.add_argument('--in_class', default=1, type=int)
    parser.add_argument('--iter', default=1, type=int)
    parser.add_argument('--train_class', default=1, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    logging.info(f'Network:\n'
                 f'\t{args.in_class} input channels\n'
                 f'\t{args.out_class} output channels (classes)\n')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    
    for train_class in [1,2,3,4]:
        for iter_idx in range(10):
            net = UNetAdvance(n_channels=args.in_class, n_classes=args.out_class, momentum=0.1)
            net = net.to(device=device)
            train_net(net=net,
                      epochs=args.epochs,
                      batch_size=args.batchsize,
                      lr=args.lr,
                      device=device,
                      img_scale=args.scale,
                      val_percent=args.val / 100,
                      iter_idx=iter_idx,
                      train_class=train_class)
