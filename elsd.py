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
from skimage import io

from datasets import landsat


def elsd(data, thresholds, old=True):
    # Returns a TxHxW matrix, where T is number of thresholds and HxW is the image dimension.
    # Hough transforms don't have any trainable parameters. We will simply do a hyperparam search
    hough_radii = np.arange(10, 20, 10)
    pred = np.zeros(data.shape)
    for cidx in range(data.shape[0]):  # Over channels
        img = np.uint8(data[cidx,:,:])
        if(old):
	        imageio.imwrite("test.pgm", img)
	        os.system("elsd test.pgm > out.txt")
	        os.system("inkscape -z -e test.png test.pgm.svg > out.txt")
        	result = io.imread("test.png", as_gray=True)
	    else:
	        imageio.imwrite("test.pgm", img)
	        os.system("elsdc test.pgm > out.txt")
	        os.system("inkscape -z -e test.png output.svg > out.txt")
        	result = io.imread("test.png", as_gray=True)
        edges = result>0
        hough_res = hough_circle(edges, hough_radii)
        for j in range(len(thresholds)):
            thr = thresholds[j]
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
        pred = elsd(data, thresholds)
        for j in range(len(thresholds)):
            accuracies[i,j] = (np.mean((pred>thresholds[j])==target))
    print(thresholds, np.mean(accuracies, axis=0))
