import os
import random
import torch.utils.data as data
from torch import distributed
import torchvision as tv
import numpy as np
from PIL import Image

classes = [
    "void",
    "Red",
    "Yellow",
    "Nattier Blue",
    "Pink",
    "Green",
    "Blue"
]


class StackingSegmentation(data.Dataset):

    def __init__(self, root, train=True, transform=None):

        root = os.path.expanduser(root)
        base_dir = "stacking"
        ade_root = os.path.join(root, base_dir)
        if train:
            split = 'training'
        else:
            split = 'validation'
        annotation_folder = os.path.join(ade_root, 'annotations', split)
        image_folder = os.path.join(ade_root, 'images', split)

        self.images = []
        fnames = sorted(os.listdir(image_folder))
        self.images = [(os.path.join(image_folder, x), os.path.join(annotation_folder, x[:-3] + "png")) for x in fnames]

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index][0]).convert('RGB')
        target = Image.open(self.images[index][1])

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.images)