"""
accuracy[0.1][8, 64, 64]:0.14253463745117187
accuracy[0.25][8, 64, 64]:0.1426849365234375
accuracy[0.5][8, 64, 64]:0.3911308288574219
accuracy[0.5][8, 64, 64]:0.3911308288574219
accuracy[0.5625][8, 64, 64]:0.7589462280273438
accuracy[0.625][8, 64, 64]:0.8495597839355469
accuracy[0.6875][8, 64, 64]:0.8568084716796875
accuracy[0.75][8, 64, 64]:0.8574653625488281
accuracy[0.9][8, 64, 64]:0.8574653625488281


"""
import os
import pdb
import numpy as np
from osgeo import gdal
gdal.UseExceptions()
import argparse
import imageio

from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle

from datasets import landsat


def hough(data, thld):
    print('Performing Hough transform...')
    # Returns a TxHxW matrix, where T is number of thresholds and HxW is the image dimension.
    thresholds = thld
    # Hough transforms don't have any trainable parameters. We will simply do a hyperparam search
    hough_radii = np.arange(10, 20, 10)
    pred = np.zeros(data.shape)
    for cidx in range(data.shape[0]):  # Over channels
        print('Channel:', cidx)
        edges = canny(data[cidx, :, :])
        hough_res = hough_circle(edges, hough_radii)
        for j in range(len(thresholds)):
            thr = thresholds[j]
            print('Threshold:', thr)
            accums, cxs, cys, radii = hough_circle_peaks(hough_res, hough_radii, threshold=thr)
            for cx, cy, r in zip(cxs, cys, radii):
                circx, circy = circle(cx, cy, r, shape=(pred.shape[0], pred.shape[1]))
                pred[cidx, circx, circy] = thr
    preds = pred.max(axis=0)
    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', dest='data_dir', type=str, default='data_small', help='Data directory.')
    args = parser.parse_args()

    dataset = landsat.SingleScene(root=args.data_dir, size=256)
    thresholds = list(np.linspace(0.5, 0.9, 5))

    accuracies = np.zeros((len(dataset), len(thresholds)))
    for i in range(len(dataset)):
        data, target, y, x = dataset[i]
        target = target>0
        pred = hough(data, thresholds)
        for j in range(len(thresholds)):
            accuracies[i,j] = (np.mean((pred>thresholds[j])==target))
    print(thresholds, np.mean(accuracies, axis=0))
