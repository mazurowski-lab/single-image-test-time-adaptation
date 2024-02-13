""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
#from attention_augmented_conv import AugmentedConv

class DoubleConvFixed(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            #nn.BatchNorm2d(mid_channels),#, track_running_stats=True),
            nn.BatchNorm2d(mid_channels, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels),#, track_running_stats=True),
            nn.BatchNorm2d(out_channels, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        
        fixed_channels = 4
        if mid_channels - fixed_channels > 0:
            self.init_conv = nn.Conv2d(in_channels, mid_channels - fixed_channels, kernel_size=3, padding=1)
        else:
            self.init_conv = None
        self.init_conv_f = nn.Conv2d(in_channels, fixed_channels, kernel_size=3, padding=1)
        with torch.no_grad():
            self.init_conv_f.bias.fill_(0.)
            self.init_conv_f.weight[0] = torch.tensor([[-1,0,1],
                                                       [-1,0,1],
                                                       [-1,0,1]])
            self.init_conv_f.weight[1] = torch.tensor([[1,0,-1],
                                                       [1,0,-1],
                                                       [1,0,-1]])
            self.init_conv_f.weight[2] = torch.tensor([[1,1,1],
                                                       [0,0,0],
                                                       [-1,-1,-1]])
            self.init_conv_f.weight[3] = torch.tensor([[-1,-1,-1],
                                                       [0,0,0],
                                                       [1,1,1]])
            #self.init_conv_f.weight[0] = torch.tensor([[-1,0,1],
            #                                           [-2,0,2],
            #                                           [-1,0,1]])
            #self.init_conv_f.weight[1] = torch.tensor([[1,0,-1],
            #                                           [2,0,-2],
            #                                           [1,0,-1]])
            #self.init_conv_f.weight[2] = torch.tensor([[1,2,1],
            #                                           [0,0,0],
            #                                           [-1,-2,-1]])
            #self.init_conv_f.weight[3] = torch.tensor([[-1,-2,-1],
            #                                           [0,0,0],
            #                                           [1,2,1]])
            #self.init_conv_f.weight[4] = torch.tensor([[-1,-1,-1],
            #                                           [-1,8,-1],
            #                                           [-1,-1,-1]])
            #self.init_conv_f.weight[7] = torch.tensor([[1,1,1],
            #                                           [1,1,1],
            #                                           [1,1,1]]) / 9.0
        
    def forward(self, x):
        if self.init_conv:
            init_1 = self.init_conv(x)
            init_2 = self.init_conv_f(x)
            x = torch.cat([init_1, init_2], 1)
        else:
            x = self.init_conv_f(x)
        return self.double_conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        #self.double_conv = nn.Sequential(
        self.cell1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels, affine=True, track_running_stats=True, momentum=1)
        #self.bn2 = nn.BatchNorm2d(mid_channels, affine=True, track_running_stats=False)
        self.cell3 = nn.ReLU(inplace=True)
        self.cell4 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True, momentum=1)
        #self.bn5 = nn.BatchNorm2d(out_channels, affine=True, track_running_stats=False)
        self.cell6 = nn.ReLU(inplace=True)

        
    def forward(self, x):
        #return self.double_conv(x)
        x = self.cell1(x)
        x = self.bn2(x)
        x = self.cell3(x)
        x = self.cell4(x)
        x = self.bn5(x)
        x = self.cell6(x)
        return x

    def get_stats(self):
        return [cell2.running_mean, cell2.running_var, cell5.running_mean, cell5.running_var]

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2=None, x3=None):
        if x2 is None:
            x = self.up(x1)
        else:
            x1 = self.up(x1)
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
            x = torch.cat([x2, x1], dim=1)

        if x3 is not None:
            # Follow x2 shape since it's from VGG
            # input is CHW
            diffY = x2.size()[2] - x3.size()[2]
            diffX = x2.size()[3] - x3.size()[3]

            x3 = F.pad(x3, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x = torch.cat([x3, x], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
