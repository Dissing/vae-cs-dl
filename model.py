import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical, Normal, kl_divergence


from modules.vae import VAE
from modules.attention import Attention


class MONet(nn.Module):

    def __init__(self, cfg):
        super(MONet, self).__init__()

        self.vae = VAE(cfg)
        self.attention = Attention()
        self.K = cfg.num_slots

        self.beta = 0.5
        self.gamma = 0.25

    def forward(self, x):

        log_s_k = [torch.zeros_like(x[:, 0:1])]

        log_m_k = []

        for i in range(self.K-1):
            log_m, log_s = self.attention(x, log_s_k[-1])
            log_s_k.append(log_s)
            log_m_k.append(log_m)
        log_m_k.append(log_s_k[-1])

        z_k, mu_k, sigma_k, x_r_k, m_r_k = self.vae(x, log_m_k)

        log_m_r_k = F.log_softmax(torch.stack(m_r_k, 4), 4)
        log_m_r_k = torch.split(log_m_r_k, 1, 4)
        log_m_r_k = [mask[:,:,:,:,0] for mask in log_m_r_k]

        masks = torch.stack(log_m_k, 4).exp()
        recons = torch.stack(x_r_k, 4)


        x_r_loss = self.reconstruct_loss(x, x_r_k, log_m_k, 0.7)
        kl_z = self.kl_loss(mu_k, sigma_k)
        kl_masks = self.attention_kl_loss(log_m_k, log_m_r_k)

        """print("x_r", x_r_loss.mean().item(),
              "kl_z", kl_z.mean().item(),
              "kl_masks", kl_masks.mean().item())"""

        loss = x_r_loss + self.beta * kl_z + self.gamma * kl_masks

        return loss, recons, masks

    def reconstruct_loss(self, x, x_r_k, log_m_k, sigma):
        loss = 0

        for x_r, log_m in zip(x_r_k, log_m_k):
            dist = Normal(x_r, sigma)
            p_x = dist.log_prob(x)
            loss += (p_x + log_m).exp()

        loss = -torch.log(loss)
        return loss.sum([1,2,3])


    def attention_kl_loss(self, log_m_k, log_m_r_k):

        mask_probs = torch.max(torch.stack(log_m_k, 4).exp(), torch.tensor(1e-5).cuda())
        predict_logits = torch.max(torch.stack(log_m_r_k, 4).exp(), torch.tensor(1e-5).cuda())


        q = Categorical(mask_probs)
        q_predict = Categorical(predict_logits)
        kl_m = kl_divergence(q, q_predict).view(log_m_k[0].shape[0], -1)
        return torch.sum(kl_m, 1)

    def kl_loss(self, mu_k, sigma_k):
        kl_z = 0

        for mu, sigma in zip(mu_k, sigma_k):
            dist = Normal(mu, sigma)
            kl_z += torch.sum(kl_divergence(dist, Normal(0.0, 1.0)), 1)
        return kl_z


class CSNet(nn.Module):

    def __init__(self, cfg):
        super(CSNet, self).__init__()

        self.vae = VAE(cfg)

        self.beta = 0.5

    def forward(self, x, m_k):

        mu_k = []
        sigma_k = []
        x_r_k = []

        for m in m_k:
            z, mu, sigma, x_r = self.vae(x, m)
            mu_k.append(mu)
            sigma_k.append(sigma)
            x_r_k.append(x_r)

        x_r_k_stack = torch.stack(x_r_k, 4)


        x_r_k_loss = self.reconstruct_loss(x, x_r_k, m_k)
        #kl_z = self.kl_loss(mu_k, sigma_k)

        """print("x_r", x_r_loss.mean().item(),
              "kl_z", kl_z.mean().item(),
              "kl_masks", kl_masks.mean().item())"""

        loss = x_r_k_loss #+ self.beta * kl_z

        return loss, x_r_k_stack

    def reconstruct_loss(self, x, x_r_k, m_k):
        loss = 0

        for x_r, m in zip(x_r_k, m_k):
            loss += F.binary_cross_entropy(x_r, x * m)

        return loss
