import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils

from skimage import io, transform
from skimage.color import rgb2hsv

class SpritesDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = os.listdir(root_dir)

    def __len__(self):
        return 4096#len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, str(idx) + ".png")
        image = io.imread(img_name)

        masks = []
        for i in range(4):
            mask_name = os.path.join(self.root_dir, str(idx) + "_" + str(i) + ".png")
            mask = io.imread(mask_name)
            masks.append(mask)


        if self.transform:
            image = self.transform(image)
            for mask in masks:
                self.transform(mask)

        return image#, masks)
