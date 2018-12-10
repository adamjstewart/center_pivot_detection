import os
import pdb
import numpy as np
from osgeo import gdal
gdal.UseExceptions()
import argparse
import imageio
from hough import hough
from elsd import elsd
from datasets import landsat
from network import Network
data_folder = 'data'
train_dataset = landsat.TimeSeries(subset='train', root=data_folder, pivots=os.path.join('data/u/sciteam/stewart1/center_pivot_detection/data',
                      'pivots_2005_utm14_{:03d}{:03d}_clipped.tif'))
print("Training Data Loaded")
# Assuming that the train dataset is given as time, channel, H, W
thresholds = list(np.linspace(0.5, 0.9, 3))
data, _, _, _, _ = train_dataset[0]
data_net_hough = np.zeros((len(train_dataset), data.shape[1], data.shape[2], data.shape[3]), dtype=np.float32)
data_net_elsd = np.zeros((len(train_dataset), data.shape[1], data.shape[2], data.shape[3]), dtype=np.float32)
target_net = np.zeros((len(train_dataset), data.shape[2], data.shape[3]))
print(len(train_dataset))
for i in range(len(train_dataset)):
    data, target, t, y, x = train_dataset[i]
    preds_hough = np.zeros((data.shape[1],data.shape[2],data.shape[3]))
    preds_elsd = np.zeros((data.shape[1],data.shape[2],data.shape[3]))
    for j in range(data.shape[1]):
        preds_hough[j,:,:] = (hough(data[:,j,:,:], thresholds))
        # preds_elsd[j,:,:] = (elsd(data[:,j,:,:], thresholds))
    data_net_hough[i,:,:,:] = preds_hough
    # data_net_elsd[i,:,:,:] = preds_elsd
    target_net[i,:,:][target>0] = 1
    print(i)
net1 = Network(data_net_hough, target_net) # Training
net2 = Network(data_net_elsd, target_net)

# val_dataset = landsat.TimeSeries(subset='val', root=data_folder, pivots=os.path.join('data/u/sciteam/stewart1/center_pivot_detection/data','pivots_2005_utm14_{:03d}{:03d}_clipped.tif'))
test_dataset = landsat.TimeSeries(subset='test', root=data_folder, pivots=os.path.join('data/u/sciteam/stewart1/center_pivot_detection/data','pivots_2005_utm14_{:03d}{:03d}_clipped.tif'))
# all_dataset = landsat.TimeSeries(subset='all', root=data_folder, pivots=os.path.join('data/u/sciteam/stewart1/center_pivot_detection/data',pivots_2005_utm14_{:03d}{:03d}_clipped.tif'))
print("Data Loaded")
data, _, _, _, _ = test_dataset[0]
test_data_net_hough = np.zeros((len(test_dataset), data.shape[1], data.shape[2], data.shape[3]))
test_data_net_elsd = np.zeros((len(test_dataset), data.shape[1], data.shape[2], data.shape[3]))
test_target_net = np.zeros((len(test_dataset), data.shape[2], data.shape[3]))

print(len(test_dataset))
accuracies = np.zeros(len(test_dataset))
for i in range(len(test_dataset)):
    data, target, t, y, x = test_dataset[i]
    preds_hough = np.zeros((data.shape[1],data.shape[2],data.shape[3]), dtype=np.float32)
    preds_elsd = np.zeros((data.shape[1],data.shape[2],data.shape[3]), dtype=np.float32)
    for j in range(data.shape[1]):
        preds_hough[j,:,:] = (hough(data[:,j,:,:], thresholds))
        # preds_elsd[j,:,:] = (elsd(data[j,:,:,:], thresholds))
    pred_max_hough = preds_hough.max(axis=0) # Max pooling
    # pred_max_elsd = preds_elsd.max(axis=0)
    print(np.mean(pred_max_elsd==target))
    accuracies[i] = np.mean(pred_max_elsd==target)
    test_data_net_hough[i,:,:,:] = preds_hough
    test_data_net_elsd[i,:,:,:] = preds_elsd
    test_target_net[i,:,:][target>0] = 1
print(np.mean(accuracies))
pred_fnn_hough = Network.score(test_data_net_hough, test_target_net) # Fully Connected Network
pred_fnn_elsd = Network.score(test_data_net_elsd, test_target_net)
print(np.mean(pred_fnn_elsd==test_target_net))
print(np.mean(pred_fnn_hough==test_target_net))