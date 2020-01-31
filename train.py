import sys
import torch
from torch import nn
from torch.nn import functional as F

from skimage import io
from skimage.color import hsv2rgb
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets.sprites import SpritesDataset

from model import MONet
from config import sprites_config

if __name__ == "__main__":

    cfg = sprites_config

    dataset = SpritesDataset("data/generated", transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available else "cpu")

    model = MONet(cfg)
    if torch.cuda.is_available():
        model = model.cuda()

    opt = torch.optim.RMSprop(model.parameters(), lr=1e-4)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[8,20,30], gamma=0.1)

    output_every = 10

    print("Initializing parameters.")
    for w in model.parameters():
        nn.init.normal_(w, mean=0.0, std=0.1)
    print("Initialized.")

    iteration = 0
    for epoch in range(1000):
        running_loss = 0.0
        for images in loader:
            v = torch.autograd.Variable(images.cuda())

            opt.zero_grad()
            batch_loss, recons, masks = model(v)

            loss = batch_loss.mean()

            loss.backward()
            opt.step()

            running_loss += loss.data.item()

            iteration += 1
            if iteration % output_every == 0:
                print("Epoch[{}/{}]-{} Loss: {:.5f}".format(epoch+1, cfg.num_epochs, iteration, running_loss/output_every))
                running_loss = 0.0
                full_reconstruction = torch.sum(masks * recons, 4)
                view = full_reconstruction.data.cpu()
                image_view = v.data.cpu()
                torchvision.utils.save_image(image_view, "image.png")
                torchvision.utils.save_image(view, "reconstruct.png")


                for i in range(4):
                    recon = recons[:,:,:,:,i].data.cpu()
                    torchvision.utils.save_image(recon, "recon%d.png" % i)
                    mask = masks[:,:,:,:,i].data.cpu()
                    torchvision.utils.save_image(mask, "mask%d.png" % i)
