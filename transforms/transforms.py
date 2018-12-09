"""PyTorch transforms for data augmentation."""

import numpy as np
import random
import torch


class ToTensor:
    """Convert a ``numpy.ndarray`` to tensor."""

    def __call__(self, pic):
        return torch.from_numpy(pic).float()


class ToNonTS:
    """Convert a ``numpy.ndarray`` to tensor."""

    def __call__(self, ncthw_array):
        c, t, h, w = ncthw_array.shape
        if isinstance(ncthw_array, np.ndarray):  # Numpy array
            return np.reshape(ncthw_array, (c * t, h, w))
        else:  # Torch tensor
            return ncthw_array.view(c * t, h, w)


class Normalize:
    """Normalize a tensor image with mean and standard deviation."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Parameters:
            tensor (Tensor): Tensor image of size (C, T, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        for band in range(tensor.shape[0]):
            tensor[band] = (tensor[band] - self.mean[band]) / self.std[band]

        return tensor


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
