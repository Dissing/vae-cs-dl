import torch
from torch import nn
from torch.nn import functional as F

from skimage import io
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets.sprites import SpritesDataset

batch_size = 64
h_dim = 576

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), h_dim // 9, 3, 3)

class VAE(nn.Module):

    def __init__(self, z_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2),
            nn.ReLU(),
            Flatten(),
        )

        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_sigma = nn.Linear(h_dim, z_dim)
        self.fc_sample = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim // 9, 32, 3, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 1, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        h = self.encoder(x)
        mu, sigma = self.fc_mu(h), self.fc_sigma(h)
        std = sigma.mul(0.5).exp_()
        eps = torch.randn(*mu.size()).cuda()
        z = mu + std * eps
        zp = self.fc_sample(z)
        return self.decoder(zp), mu, sigma

def loss_fn(rx, x, mu, sigma):
    BCE = F.mse_loss(rx, x, size_average=False)

    KLD = -0.5 * torch.sum(1 + sigma - mu**2 - sigma.exp())
    return BCE + 1 * KLD

if __name__ == "__main__":

    dataset = SpritesDataset("data/generated", transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available else "cpu")

    model = VAE(15)
    if torch.cuda.is_available():
        model.cuda()

    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[8,20], gamma=0.1)

    epochs = 200

    for epoch in range(epochs):
        for idx, images in enumerate(loader):
            v = torch.autograd.Variable(images.cuda())
            reconstructs, mu, sigma = model(v)
            loss = loss_fn(reconstructs, v, mu, sigma)


            opt.zero_grad()
            loss.backward()
            opt.step()

            if idx % 100 == 0:
                print("Epoch[{}/{}] Loss: {:.5f}".format(epoch+1, epochs, loss.data.item()))
                view = reconstructs.data.cpu()
                image_view = v.data.cpu()
                torchvision.utils.save_image(image_view, "image.png")
                torchvision.utils.save_image(view, "reconstruct.png")
        scheduler.step()
