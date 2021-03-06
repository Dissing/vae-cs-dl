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
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, str(idx) + ".png")

        image = io.imread(img_name)

        #image = np.float32(rgb2hsv(image))

        if self.transform:
            image = self.transform(image)

        return image
