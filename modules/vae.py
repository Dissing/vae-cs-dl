
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

from modules import blocks

class Encoder(nn.Module):

    def __init__(self, cfg):
        super(Encoder, self).__init__()

        c = cfg.encoder_channels
        z = cfg.latent_dimensions

        mlp_in = 2 * c * (cfg.img_size // 16) ** 2
        mlp_hidden = 256
        mlp_out = 2 * z

        self.encoder = nn.Sequential(
            nn.Conv2d(4, c, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(c, c, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(c, 2*c, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(2*c, 2*c, 3, 2, 1),
            nn.ReLU(),
            blocks.Flatten(),
            nn.Linear(mlp_in, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_out)
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):

    def __init__(self, cfg):
        super(Decoder, self).__init__()

        c = cfg.decoder_channels
        z = cfg.latent_dimensions

        self.decoder = nn.Sequential(
            blocks.BroadcastLayer(cfg.img_size + 8),
            nn.Conv2d(z+2, c, 3),
            nn.ReLU(),
            nn.Conv2d(c, c, 3),
            nn.ReLU(),
            nn.Conv2d(c, c, 3),
            nn.ReLU(),
            nn.Conv2d(c, c, 3),
            nn.ReLU(),
            nn.Conv2d(c, 4, 1),
        )

    def forward(self, x):
        return self.decoder(x)



class VAE(nn.Module):

    def __init__(self, cfg):
        super(VAE, self).__init__()

        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)


    def forward(self, x, masks):

        # Encoding

        K = len(masks)

        x = x.repeat(K, 1, 1, 1)
        masks = torch.cat(masks, dim=0)

        encoder_input = torch.cat((masks, x), dim=1)
        encoder_output = self.encoder(encoder_input)
        mu, sigma = torch.chunk(encoder_output, 2, dim=1)

        sigma = F.softplus(sigma + 0.5) + 1e-8

        # Sampling

        dist = Normal(mu, sigma)
        z = dist.rsample()

        # Decoding

        decoder_outputs = self.decoder(z)
        x_m_r_k = torch.chunk(torch.sigmoid(decoder_outputs), K, 0)
        z_k = torch.chunk(z, K, 0)
        mu_k = torch.chunk(mu, K, 0)
        sigma_k = torch.chunk(sigma, K, 0)
        x_r_k = [x_m_r[:, :3] for x_m_r in x_m_r_k]
        m_r_k = [x_m_r[:, 3:] for x_m_r in x_m_r_k]

        return z_k, mu_k, sigma_k, x_r_k, m_r_k
