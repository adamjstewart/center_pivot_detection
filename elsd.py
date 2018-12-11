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
elsd_folder = "./ELSD"
elsdc_folder = "./ELSDc"

from datasets import landsat
def elsd(data, thld, old=True):
    # Returns a TxHxW matrix, where T is number of thresholds and HxW is the image dimension.
    thresholds = thld
    # Hough transforms don't have any trainable parameters. We will simply do a hyperparam search
    hough_radii = np.arange(10, 20, 10)
    pred = np.zeros(data.shape)
    for cidx in range(data.shape[0]):  # Over channels
        img = np.uint8(data[cidx,:,:])
        if(old):
	        imageio.imwrite(elsd_folder+"/test.pgm", img)
	        os.system(elsd_folder+"/elsd "+elsd_folder+"/test.pgm > out.txt")
	        os.system("inkscape -z -e "+elsd_folder+"/test.png "+elsd_folder+"/test.pgm.svg > out.txt")
        	result = io.imread(elsd_folder+"/test.png", as_gray=True)
	    else:
	        imageio.imwrite(elsdc_folder+"/test.pgm", img)
	        os.system(elsdc_folder+"/elsdc "+elsdc_folder+"/test.pgm > out.txt")
	        os.system("inkscape -z -e "+elsdc_folder+"/test.png "+elsdc_folder+"/output.svg > out.txt")
        	result = io.imread(elsdc_folder+"/test.png", as_gray=True)
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