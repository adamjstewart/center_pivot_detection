"""PyTorch Dataset for loading Landsat data."""

from osgeo import gdal
from torch.utils.data import Dataset

import glob
import numpy as np
import os


class SingleScene(Dataset):
    """This Dataset loads subsets of a single scene."""

    def __init__(self, root='data', size=128, stride=64,
                 transform=None, target_transform=None):
        """Initializes a new Dataset.

        Parameters:
            root (str, optional): the root directory containing the scene
            size (int, optional): the height and width of each subset
            stride (int, optional): the stride with which to divide subsets
            transform (callable, optional): a function/transform that takes in
                a numpy array and returns a transformed version
            target_transform (callable, optional): a function/transform that
                takes in the target and transforms it
        """
        self.root = root
        self.size = size
        self.stride = stride
        self.transform = transform
        self.target_transform = target_transform

        # Load the scene
        scene = []
        for band in [1, 2, 3, 4, 5, 7]:
            filename = '*_sr_band{}_clipped.tif'.format(band)
            filename = glob.glob(os.path.join(root, filename))[0]
            ds = gdal.Open(filename)
            ar = ds.ReadAsArray()
            scene.append(ar)
        self.scene = np.array(scene)

        # Load the segmentation
        filename = glob.glob(os.path.join(root, 'pivots_*_clipped.tif'))[0]
        ds = gdal.Open(filename)
        self.dataset = ds
        self.segmentation = ds.ReadAsArray()

    def __len__(self):
        """The size of the Dataset.

        Returns:
            int: the number of subsets of the image
        """
        return self.width * self.height

    def __getitem__(self, idx):
        """An individual subset of the image.

        Parameters:
            idx (int): the index to return

        Returns:
            numpy.ndarray: a 3D tensor of raw satellite imagery
            numpy.ndarray: the correct segmentation of the image
            int: the starting row the subset was extracted from
            int: the starting col the subset was extracted from
        """
        # Border often contains invalid pixels.
        # Center the subsets within the scene.
        y_offset = (self.scene.shape[1] - (self.size * self.height)) // 2
        x_offset = (self.scene.shape[2] - (self.size * self.width)) // 2

        # Convert the index to a (row, col) location
        row = idx // self.width
        col = idx % self.width

        # Find the exact coordinates in the array
        y = row * self.size + y_offset
        x = col * self.size + x_offset

        data = self.scene[:, y:y + self.size, x:x + self.size]
        target = self.segmentation[y:y + self.size, x:x + self.size]

        # Apply any requested transforms
        if self.transform:
            data = self.transform(data)

        if self.target_transform:
            target = self.target_transform(target)

        return data, target, y, x

    @property
    def height(self):
        """The height of the scene.

        Returns:
            int: the number of subsets high the scene is
        """
        return (self.scene.shape[1] - self.size) // self.size + 1

    @property
    def width(self):
        """The width of the scene.

        Returns:
            int: the number of subsets wide the scene is
        """
        return (self.scene.shape[2] - self.size) // self.size + 1

    def write(self, predictions, filename):
        """Write a set of predictions to a GeoTIFF file.

        Parameters:
            predictions (list): a list of (prediction, y, x) tuples
            filename (str): the output filename
        """
        driver = self.dataset.GetDriver()
        dst_ds = driver.CreateCopy(filename, self.dataset)

        prediction_array = np.zeros_like(self.segmentation)
        for prediction, y, x in predictions:
            prediction_array[y:y + self.size, x:x + self.size] = prediction

        # Overwrite the raster band with the predicted labels
        band = dst_ds.GetRasterBand(1)
        band.WriteArray(prediction_array)
