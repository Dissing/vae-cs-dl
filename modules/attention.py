import torch
from torch import nn
from torch.nn import functional as F

from modules.unet import UNet

class Attention(nn.Module):

    def __init__(self):
        super(Attention, self).__init__()
        self.unet = UNet()

    def forward(self, x, scope):

        logits = self.unet(torch.cat((x, scope), 1))

        log_alpha = F.logsigmoid(logits)
        log_neg_alpha = F.logsigmoid(-logits)

        log_m = scope + log_alpha

        log_s = scope + log_neg_alpha

        return log_m, log_s

