import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):

    def __init__(self, dim):
        super(UnFlatten, self).__init__()
        self.dim = dim

    def forward(self, input):
        return input.view(*self.dim)


class BroadcastLayer(nn.Module):

    def __init__(self, dim):
        super(BroadcastLayer, self).__init__()
        self.dim = dim

        x_space = torch.linspace(-1, 1, dim)
        y_space = torch.linspace(-1, 1, dim)

        x_grid, y_grid = torch.meshgrid(x_space, y_space)

        self.register_buffer('x_grid', x_grid.view((1,1) + x_grid.shape))
        self.register_buffer('y_grid', y_grid.view((1,1) + y_grid.shape))

    def forward(self, x):
        x = x.view(x.size(0), -1, 1, 1).expand(-1, -1, self.dim, self.dim)
        xg = self.x_grid.expand(x.size(0), -1, -1, -1)
        yg = self.y_grid.expand(x.size(0), -1, -1, -1)
        return torch.cat((x, xg, yg), dim=1)
