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

from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle

from datasets import landsat

parser = argparse.ArgumentParser()
parser.add_argument('--d', dest='data_dir', type=str, default='data/baby_data', help='Data directory.')
args = parser.parse_args()


dataset = landsat.SingleScene(root=args.data_dir, size=256)

for (min_r, max_r, nr) in [(8, 64, 64)]:
    thresholds = list(np.linspace(0.1, 0.9, 20))
    print("min_r:{}, max_r:{}, nr:{}, thresholds:{}".format(min_r, max_r, nr, thresholds))
    # Hough transforms don't have any trainable parameters. We will simply do a hyperparam search
    hough_radii = np.arange(min_r, max_r, nr)

    accuracies = {thr: np.zeros(len(dataset), dtype=np.float) for thr in thresholds}
    for i in range(len(dataset)):  # Over datapoints
        data, target, y, x = dataset[i]
        target = target > 0

        pred = {thr: np.zeros(data.shape, dtype=bool) for thr in thresholds}

        for cidx in range(data.shape[0]):  # Over channels
            edges = canny(data[cidx, :, :])
            hough_res = hough_circle(edges, hough_radii)
            for thr in thresholds:
                accums, cxs, cys, radii = hough_circle_peaks(hough_res, hough_radii, threshold=thr)
                for cx, cy, r in zip(cxs, cys, radii):
                    circx, circy = circle(cx, cy, r, shape=(pred[thr].shape[1], pred[thr].shape[2]))
                    pred[thr][cidx, circx, circy] = True

        # Compute accuracy
        for thr in thresholds:
            accuracies[thr][i] = np.mean(target == (pred[thr].max(axis=0)))  # Simple max pool over channels
    best_thr = None
    for thr in thresholds:
        if ((best_thr is None) or (np.mean(accuracies[thr]) > np.mean(accuracies[best_thr]))):
            best_thr = thr
        print("accuracy[{}][{}, {}, {}]:{}".format(thr, min_r, max_r, nr, np.mean(accuracies[thr])))
    print("best: accuracy[{}][{}, {}, {}]:{}".format(best_thr, min_r, max_r, nr, np.mean(accuracies[best_thr])))
