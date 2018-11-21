import os
import pdb
import numpy as np
from osgeo import gdal
gdal.UseExceptions()
import argparse

from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle

from datasets import landsat

parser = argparse.ArgumentParser()
parser.add_argument('--d', dest='data_dir', type=str, default='data/baby_data', help='Data directory.')
args = parser.parse_args()


dataset = landsat.SingleScene(root=args.data_dir, size=256)

for (min_r, max_r, nr) in [(10, 32, 10)]:
    for threshold in [0.5]:
        print("min_r:{}, max_r:{}, nr:{}, threshold:{}".format(min_r, max_r, nr, threshold))
        # Hough transforms don't have any trainable parameters. We will simply do a hyperparam search
        hough_radii = np.arange(min_r, max_r, nr)
        threshold = 0.1

        accuracies = np.zeros(len(dataset), dtype=np.float)
        for i in range(len(dataset)):  # Over datapoints
            data, target, y, x = dataset[i]
            target = target > 0

            pred = np.zeros(data.shape, dtype=bool)

            for cidx in range(data.shape[0]):  # Over channels
                edges = canny(data[cidx, :, :])
                hough_res = hough_circle(edges, hough_radii)
                accums, cxs, cys, radii = hough_circle_peaks(hough_res, hough_radii, threshold=threshold)
                for cx, cy, r in zip(cxs, cys, radii):
                    circx, circy = circle(cx, cy, r, shape=(pred.shape[1], pred.shape[2]))
                    pred[cidx, circx, circy] = True

            # Compute accuracy
            accuracies[i] = np.mean(target == (pred.max(axis=0)))  # Simple max pool over channels

        print("accuracy[{}, {}, {}, {}]:{}".format(min_r, max_r, nr, threshold, np.mean(accuracies)))
