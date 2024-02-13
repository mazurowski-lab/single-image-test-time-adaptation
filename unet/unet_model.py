""" Full assembly of the parts to form the complete network """
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import functools

import cv2
import math
import numpy as np
from .unet_parts import *

def adain(f, ft, p=False):
    assert len(f.shape) == 4
    assert len(ft.shape) == 4
    ft = ft.detach()

    #return f, torch.tensor([-1])
    
    if 0:
        f_mean, f_std = f.mean(), f.std()
        f_norm = (f-f_mean) / f_std
        ft_mean, ft_std = ft.mean(), ft.std()
        #diff = (f_mean - ft_mean).mean([1,2,3])
        diff = f.mean([1,2,3]) - ft_mean
        return ft_std * f_norm + ft_mean
    else:
        ft_mean, ft_std = ft.mean([2,3]), ft.std([2,3])
        #f_norm = F.instance_norm(f, running_mean=ft_mean, running_var=ft_std)
        
        f_mean, f_std = f.mean([2,3]), f.std([2,3])
        f_std = f_std + 1e-10
        f_norm = (f.permute([2,3,0,1]) - f_mean) / f_std 
        f_norm = f_norm * ft_std + ft_mean
        f_norm = f_norm.permute([2,3,0,1])
        return f_norm


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, stochastic=False, fixed=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.stochastic = stochastic
        
        mid_channel = 64
        self.inc = DoubleConv(n_channels, mid_channel)
        self.down1 = Down(mid_channel, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(64+mid_channel, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def adain(self, f, ft, p=False):
        assert len(f.shape) == 4
        assert len(ft.shape) == 4
        ft = ft.detach()

        #return f, torch.tensor([-1])
        
        if 0:
            f_mean, f_std = f.mean(), f.std()
            f_norm = (f-f_mean) / f_std
            ft_mean, ft_std = ft.mean(), ft.std()
            #diff = (f_mean - ft_mean).mean([1,2,3])
            diff = f.mean([1,2,3]) - ft_mean
            return ft_std * f_norm + ft_mean, diff
        else:
            ft_mean, ft_std = ft.mean([2,3]), ft.std([2,3])
            #f_norm = F.instance_norm(f, running_mean=ft_mean, running_var=ft_std)
            
            f_mean, f_std = f.mean([2,3]), f.std([2,3])
            f_std = f_std + 1e-10
            f_norm = (f.permute([2,3,0,1]) - f_mean) / f_std 
            f_norm = f_norm * ft_std + ft_mean
            f_norm = f_norm.permute([2,3,0,1])
            return f_norm, torch.tensor([-1])

    def forward(self, x, x_t=None, p=False):
        x1 = self.inc(x)
        if x_t is not None:
            x1_t = self.inc(x_t)
            x1, diff1 = self.adain(x1, x1_t, p=p)
        x2 = self.down1(x1)
        if x_t is not None:
            x2_t = self.down1(x1_t)
            x2, diff2 = self.adain(x2, x2_t)
        x3 = self.down2(x2)
        if x_t is not None:
            x3_t = self.down2(x2_t)
            x3, diff3 = self.adain(x3, x3_t)
        x4 = self.down3(x3)
        if x_t is not None:
            x4_t = self.down3(x3_t)
            x4, diff4 = self.adain(x4, x4_t)
        x5 = self.down4(x4)
        if x_t is not None:
            x5_t = self.down4(x4_t)
            x5, diff5 = self.adain(x5, x5_t)

        #if x_t:
        #    x1_t = self.inc(x_t)
        #    x2_t = self.down1(x1_t)
        #    x3_t = self.down2(x2_t)
        #    x4_t = self.down3(x3_t)
        #    x5_t = self.down4(x4_t)

        x = self.up1(x5, x4)
        #if x_t is not None:
        #    x_t = self.up1(x5_t, x4_t)
        #    x, _ = self.adain(x, x_t)
        x = self.up2(x, x3)
        #if x_t is not None:
        #    x_t = self.up2(x_t, x3_t)
        #    x, _ = self.adain(x, x_t)
        x = self.up3(x, x2)
        #if x_t is not None:
        #    x_t = self.up3(x_t, x2_t)
        #    x, _ = self.adain(x, x_t)
        x = self.up4(x, x1)
        #if x_t is not None:
        #    x_t = self.up4(x_t, x1_t)
        #    x, _ = self.adain(x, x_t)
        logits = self.outc(x)

        return logits

class UNetTwoBranch(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, stochastic=False, fixed=False):
        super(UNetTwoBranch, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.stochastic = stochastic
        
        mid_channel = 64
        self.inc = DoubleConv(n_channels, mid_channel)
        self.down1 = Down(mid_channel, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(64+mid_channel, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.re_up1 = Up(512, 512, bilinear)
        self.re_up2 = Up(512, 256, bilinear)
        self.re_up3 = Up(256, 128, bilinear)
        self.re_up4 = Up(64+mid_channel, 64, bilinear)
        self.re_outc = OutConv(64, n_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        re_x = self.re_up1(x5)
        re_x = self.re_up2(re_x)
        re_x = self.re_up3(re_x)
        re_x = self.re_up4(re_x)
        recons = self.re_outc(re_x)

        return logits, recons

# New Model
class UNetAdvance(nn.Module):
    # static variable
    mean_tracker = []
    std_tracker = []

    def __init__(
        self, 
        n_channels, 
        n_classes, 
        model_channels=64,
        channel_mult=(1,2,4,8), 
        attention_resolutions=(8,16),
        num_res_blocks=2,
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=4,
        dropout=0.1,
        momentum=1
    ):
        super(UNetAdvance, self).__init__()

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([nn.Conv2d(n_channels, ch, 3, padding=1)])
        self.dtype = torch.float32

        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(
                                ch,
                                dropout,
                                out_channels=int(mult * model_channels),
                                momentum=momentum,
                )]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels, 
                            momentum=momentum))
                self.input_blocks.append(Sequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(Downsample(ch, True, out_channels=out_ch))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = Sequential(
            ResBlock(
                ch,
                dropout,
                momentum=momentum,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                momentum=momentum,
            ),
            ResBlock(
                ch,
                dropout,
                momentum=momentum,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        dropout,
                        out_channels=int(model_channels * mult),
                        momentum=momentum,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            momentum=momentum,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        Upsample(ch, True, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(Sequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch, momentum=momentum),
            nn.SiLU(),
            zero_module(nn.Conv2d(input_ch, n_classes, 3, padding=1)),
        )

    def forward(self, x, x_val=None):
        hs = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h)
            hs.append(h)
        h = self.middle_block(h)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h)
        return self.out(h)

# Helper Func.
class Sequential(nn.Sequential):
    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

#def forward_hook(module, input, output):
#    UNetAdvance.mean_tracker.append(module.running_mean.cpu().clone())
#    UNetAdvance.std_tracker.append(module.running_var.cpu().clone())

class MyBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super(MyBatchNorm2d, self).__init__(*args, **kwargs)

    def forward(self, input):
        running_mean_temp = input.mean(dim=(0, 2, 3))
        running_var_temp = input.var(dim=(0, 2, 3))
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * running_mean_temp
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * running_var_temp

        return super().forward(input)

class MyBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super(MyBatchNorm1d, self).__init__(*args, **kwargs)

    def forward(self, input):
        running_mean_temp = input.mean(dim=(0, 2))
        running_var_temp = input.var(dim=(0, 2))
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * running_mean_temp
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * running_var_temp

        return super().forward(input)


def normalization(channels, dim=2, momentum=1):
    print('Init w/ momentum', momentum)
    if momentum != 0.8:
        if dim==2:
            ret = nn.BatchNorm2d(channels, track_running_stats=True, momentum=momentum)
        if dim==1:
            ret = nn.BatchNorm1d(channels, track_running_stats=True, momentum=momentum)
        return ret
    else:
        print('Custom BN')
        if dim==2:
            ret = MyBatchNorm2d(channels, track_running_stats=True, momentum=momentum)
        if dim==1:
            ret = MyBatchNorm1d(channels, track_running_stats=True, momentum=momentum)
        return ret


    ## Training model
    #if 0:
    #    if dim==2:
    #        ret = nn.BatchNorm2d(channels, track_running_stats=True, momentum=0.1)
    #    if dim==1:
    #        ret = nn.BatchNorm1d(channels, track_running_stats=True, momentum=0.1)
    ## Eval model
    #elif 0:
    #    if dim==2:
    #        ret = nn.BatchNorm2d(channels, track_running_stats=True, momentum=1)
    #    if dim==1:
    #        ret = nn.BatchNorm1d(channels, track_running_stats=True, momentum=1)
    #    #ret.register_forward_hook(forward_hook)
    ## CBNA
    #else:
    #    if dim==2:
    #        ret = nn.BatchNorm2d(channels, track_running_stats=True, momentum=0.2)
    #    if dim==1:
    #        ret = nn.BatchNorm2d(channels, track_running_stats=True, momentum=0.2)

    return ret

class Upsample(nn.Module):
    def __init__(self, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        stride = 2 
        if use_conv:
            self.op = nn.Conv2d(self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class ResBlock(nn.Module):
    def __init__(
        self,
        channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_checkpoint=False,
        up=False,
        down=False,
        momentum=0.1,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            normalization(channels, momentum=momentum),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.out_layers = nn.Sequential(
            normalization(self.out_channels, momentum=momentum),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_new_attention_order=False,
        encoder=False,
        momentum=0.1,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = normalization(channels, 1, momentum=momentum)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

class QKVAttentionLegacy(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

#-----------------
# UNet from paper: On-the-fly-adaptation
class UNetVGG(nn.Module):
    def __init__(self, n_channels, n_classes, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256,512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(n_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out
