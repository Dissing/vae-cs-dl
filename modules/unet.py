
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.blocks import Flatten

class NormalizedConv2d(nn.Module):

    def __init__(self, nin, nout):
        super(NormalizedConv2d, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(nin, nout, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(nout, affine=True),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        c = 32

        self.downscale = nn.ModuleList([
            NormalizedConv2d(4, c),
            NormalizedConv2d(c, c),
            NormalizedConv2d(c, 2*c),
            NormalizedConv2d(2*c, 2*c),
            NormalizedConv2d(2*c, 2*c),
        ])

        self.upscale = nn.ModuleList([
            NormalizedConv2d(4*c, 2*c),
            NormalizedConv2d(4*c, 2*c),
            NormalizedConv2d(4*c, c),
            NormalizedConv2d(2*c, c),
            NormalizedConv2d(2*c, c),
        ])

        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(32*c, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 32*c),
            nn.ReLU(),
        )

        self.final_conv = nn.Conv2d(c, 1, 1)

    def forward(self, x):

        down = x
        skip = []

        for i, block in enumerate(self.downscale):
            act = block(down)
            skip.append(act)
            if i < 4:
                act = F.interpolate(act, scale_factor=0.5, mode='nearest')
            down = act

        up = self.mlp(down).view(x.size(0), -1, 4, 4)

        for i, block in enumerate(self.upscale):
            up = block(torch.cat([up, skip[-1 - i]], 1))
            if i < 4:
                up = F.interpolate(up, scale_factor=2.0, mode='nearest')

        return self.final_conv(up)
