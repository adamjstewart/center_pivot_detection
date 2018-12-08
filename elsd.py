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
from imageio import imread
from imageio import imwrite

from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle

from datasets import landsat

parser = argparse.ArgumentParser()
parser.add_argument('--d', dest='data_dir', type=str, default='./data', help='Data directory.')
args = parser.parse_args()


dataset = landsat.SingleScene(root=args.data_dir, size=256)
accuracy = np.zeros(len(dataset))

for i in range(len(dataset)):  # Over datapoints
    data, target, y, x = dataset[i]
    result = np.zeros(data.shape)
    print(data.shape)
    print(target.shape)
    for j in range(data.shape[0]):
        print(data[j,:,:])
        imwrite("./ELSD/test.pgm", data[j,:,:])
        os.system("./ELSD/elsd ./ELSD/test.pgm")
        result[j,:,:] = Segment("./ELSD/test.pgm")
    accuracy[i] = np.mean(result, target)

print(np.mean(accuracy))