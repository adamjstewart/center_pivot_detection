"""PyTorch transforms for data augmentation."""

import numpy as np
import random
import torch


class ToTensor:
    """Convert a ``numpy.ndarray`` to tensor."""

    def __call__(self, pic):
        return torch.from_numpy(pic).float()


class RandomHorizontalFlip:
    """Horizontally flip the given ``numpy.ndarray`` randomly
    with a given probability."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Parameters:
            img (numpy.ndarray): Image to be flipped

        Returns:
            numpy.ndarray: Randomly flipped image
        """
        if random.random() < self.p:
            return np.flipud(img)
        return img


class RandomVerticalFlip:
    """Vertically flip the given ``numpy.ndarray`` randomly
    with a given probability."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Parameters:
            img (numpy.ndarray): Image to be flipped

        Returns:
            numpy.ndarray: Randomly flipped image
        """
        if random.random() < self.p:
            return np.fliplr(img)
        return img
