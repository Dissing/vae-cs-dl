import torch
from torch import nn
from torch.nn import functional as F

from skimage import io
from skimage.color import hsv2rgb
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets.sprites import SpritesDataset

batch_size = 64
f_depth = 64
f_width = 8
h_dim = f_depth * f_width * f_width

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), f_depth, f_width, f_width)

class VAE(nn.Module):

    def __init__(self, z_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #nn.Conv2d(32, 64, 3, stride=2, padding=1),
            #nn.BatchNorm2d(64),
            #nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            Flatten(),
        )

        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_sigma = nn.Linear(h_dim, z_dim)
        self.fc_sample = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(f_depth, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1),
            #nn.BatchNorm2d(32),
            #nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 1, 1),
            nn.Sigmoid(),
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
    BCE = F.binary_cross_entropy(rx, x, reduction='sum')

    KLD = -0.5 * torch.sum(1 + sigma - mu**2 - sigma.exp())
    return BCE + 2.0 * KLD

if __name__ == "__main__":

    dataset = SpritesDataset("data/generated", transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available else "cpu")

    model = VAE(10)
    if torch.cuda.is_available():
        model.cuda()

    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[8,20,30], gamma=0.1)

    epochs = 50

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
