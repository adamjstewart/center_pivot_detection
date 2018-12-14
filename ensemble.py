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
from sklearn import metrics

def test(prediction_array, target_array, prefix):
    # Flatten arrays
    prediction_array = prediction_array.flatten()
    target_array = target_array.flatten()

    # Compute statistics
    conf_mat = metrics.confusion_matrix(target_array, prediction_array)
    acc = metrics.accuracy_score(target_array, prediction_array)
    prec = metrics.precision_score(target_array, prediction_array)
    rec = metrics.recall_score(target_array, prediction_array)
    f1 = metrics.f1_score(target_array, prediction_array)
    kappa = metrics.cohen_kappa_score(target_array, prediction_array)

    # Print statistics
    print('{} net conf_matrix: \n{}'.format(prefix, conf_mat))
    print('{} net accuracy: {}'.format(prefix, acc))
    print('{} net precision: {}'.format(prefix, prec))
    print('{} net recall: {}'.format(prefix, rec))
    print('{} net f1: {}'.format(prefix, f1))
    print('{} net kappa: {}'.format(prefix, kappa))

parser = argparse.ArgumentParser()
parser.add_argument('-t', "--type", dest='model_type', type=str, default='canny', help='CV model to use.')
args = parser.parse_args()
train_dataset = landsat.TimeSeries(subset='train')
print("Training Data Loaded")
# Assuming that the train dataset is given as channel, time, H, W
thresholds = list(np.linspace(0.1, 0.9, 5))
data, _, _, _, _ = train_dataset[0]
data_net = np.zeros((len(train_dataset), data.shape[1], data.shape[2], data.shape[3]), dtype=np.float32)
target_net = np.zeros((len(train_dataset), data.shape[2], data.shape[3]))
for i in range(len(train_dataset)):
    data, target, t, y, x = train_dataset[i]
    preds = np.zeros((data.shape[1],data.shape[2],data.shape[3]))
    for j in range(data.shape[1]):
        if(args.model_type=='canny'):
            preds[j,:,:] = (hough(data[:,j,:,:], thresholds))
        if(args.model_type=='elsd'):
            preds[j,:,:] = (elsd(data[:,j,:,:], thresholds, old=True))
        if(args.model_type=='elsdc'):
            preds[j,:,:] = (elsd(data[:,j,:,:], thresholds, old=False))
    data_net[i,:,:,:] = preds
    target_net[i,:,:][target>0] = 1
print("Training Network")
net1 = Network(data_net, target_net) # Training

test_dataset = landsat.TimeSeries(subset='test')
print("Test Data Loaded")
data, _, _, _, _ = test_dataset[0]
test_data_net = np.zeros((len(test_dataset), data.shape[1], data.shape[2], data.shape[3]))
test_target_net = np.zeros((len(test_dataset), data.shape[2], data.shape[3]))

accuracies = np.zeros(len(test_dataset))
for i in range(len(test_dataset)):
    data, target, t, y, x = test_dataset[i]
    preds = np.zeros((data.shape[1],data.shape[2],data.shape[3]), dtype=np.float32)
    for j in range(data.shape[1]):
        if(args.model_type=='canny'):
            preds[j,:,:] = (hough(data[:,j,:,:], thresholds))
        if(args.model_type=='elsd'):
            preds[j,:,:] = (elsd(data[:,j,:,:], thresholds, old=True))
        if(args.model_type=='elsdc'):
            preds[j,:,:] = (elsd(data[:,j,:,:], thresholds, old=False))
    pred_max = preds.max(axis=0) # Max pooling
    accuracies[i] = np.mean(pred_max==(target>0))
    test_data_net[i,:,:,:] = preds
    test_target_net[i,:,:] = target>0
print(np.mean(accuracies))
pred_fnn = net1.test(data_net) # Fully Connected Network
train_dataset.write(pred_fnn>0.5, 'train_'+args.model_type+'_predictions.tif')
test((pred_fnn>0.5),target_net, 'train')
pred_fnn_test = net1.test(test_data_net) # Fully Connected Network
test_dataset.write(pred_fnn_test>0.5, 'test_'+args.model_type+'_predictions.tif')
test((pred_fnn_test>0.5),test_target_net, 'test')
