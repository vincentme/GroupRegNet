import torch
from torch import nn
import torch.nn.functional as F
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class UNet(nn.Module):
    '''
    U-net implementation with modifications. 
        1. Works for input of 2D or 3D
        2. Change batch normalization to instance normalization
    
    Adapted from https://github.com/jvanvugt/pytorch-unet/blob/master/unet.py
    

    Parameters
    ----------
    in_channels : int
        number of input channels.
    out_channels : int
        number of output channels.
    dim : (2 or 3), optional
        The dimention of input data. The default is 2.
    depth : int, optional
        Depth of the network. The maximum number of channels will be 2**(depth - 1) times than the initial_channels. The default is 5.
    initial_channels : TYPE, optional
        Number of initial channels. The default is 32.
    normalization : bool, optional
        Whether to add instance normalization after activation. The default is False.
    '''
    def __init__(self, in_channels, out_channels, dim = 2, depth = 5, initial_channels = 32, normalization = True):

        super().__init__()
        assert dim in (2, 3)
        self.dim = dim

        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(self.depth):
            current_channels = 2**i*initial_channels
            self.down_path.append(ConvBlock(prev_channels, current_channels, dim, normalization))
            prev_channels = current_channels

        self.up_path = nn.ModuleList()
        for i in reversed(range(self.depth - 1)):
            current_channels = 2**i*initial_channels
            # print(prev_channels, current_channels)
            self.up_path.append(UpBlock(prev_channels, current_channels, dim, normalization))
            prev_channels = current_channels

        if dim == 2:
            self.last = nn.Conv2d(prev_channels, out_channels, kernel_size = 1)
        elif dim == 3:
            self.last = nn.Conv3d(prev_channels, out_channels, kernel_size = 1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i < self.depth - 1:
                blocks.append(x)
                x = F.interpolate(x, scale_factor = 0.5, mode = 'bilinear' if self.dim == 2 else 'trilinear', align_corners = True, recompute_scale_factor = False)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim, normalization, LeakyReLU_slope = 0.2):
        super().__init__()
        block = []
        if dim == 2:
            block.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding = 1))
            if normalization:
                block.append(nn.InstanceNorm2d(out_channels))
            block.append(nn.LeakyReLU(LeakyReLU_slope))
        elif dim == 3:
            block.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding = 1))
            if normalization:
                block.append(nn.InstanceNorm3d(out_channels))
            block.append(nn.LeakyReLU(LeakyReLU_slope))
        else:
            raise (f'dim should be 2 or 3, got {dim}')
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim, normalization):
        super().__init__()
        self.dim = dim
        if dim == 2:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1)
        elif dim == 3:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size = 1)
        self.conv_block = ConvBlock(in_channels, out_channels, dim, normalization)


    def forward(self, x, skip):
        x_up = F.interpolate(x, skip.shape[2:], mode = 'bilinear' if self.dim == 2 else 'trilinear', align_corners = True)
        x_up_conv = self.conv(x_up)
        out = torch.cat([x_up_conv, skip], 1)
        out = self.conv_block(out)
        return out
    
