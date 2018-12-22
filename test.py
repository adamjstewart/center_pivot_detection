#!/usr/bin/env python3

from osgeo import gdal
from sklearn import metrics

import argparse
import numpy as np
import os


def set_up_parser():
    """Set up the argument parser.

    Returns:
        argparse.ArgumentParser: the argument parser
    """
    # Initialize new parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--ground-truth',
        default='data/pivots_2005/pivots_2005_utm14_029032_clipped.tif',
        help='ground truth labels')
    parser.add_argument(
        '--prediction-dir',
        default='results')

    return parser


def calculate_statistics(truth_array, prediction_array):
    """Calculates the statistics of our data.

    Arrays must have the same shape.

    Parameters:
        truth_array (np.ndarray): the ground truth array
        prediction_array (np.ndarray): array containing predicted labels
    """
    assert truth_array.shape == prediction_array.shape

    # Flatten arrays
    truth_array = truth_array.flatten()
    prediction_array = prediction_array.flatten()

    # Compute and print statistics
    conf_mat = metrics.confusion_matrix(truth_array, prediction_array)
    print('conf_matrix: \n{}'.format(conf_mat))
    acc = metrics.accuracy_score(truth_array, prediction_array)
    print('accuracy: {}'.format(acc))
    prec = metrics.precision_score(truth_array, prediction_array)
    print('precision: {}'.format(prec))
    rec = metrics.recall_score(truth_array, prediction_array)
    print('recall: {}'.format(rec))
    f1 = metrics.f1_score(truth_array, prediction_array)
    print('f1: {}'.format(f1))
    kappa = metrics.cohen_kappa_score(truth_array, prediction_array)
    print('kappa: {}'.format(kappa))


def compute_subsets(truth_file, prediction_file):
    """Calculates the statistics of each possible subset of data.

    Parameters:
        truth_file (str): file containing the ground truth labels
        prediction_file (str): file containing the predicted labels
    """
    print('\nFile:', prediction_file)

    truth_ds = gdal.Open(truth_file)
    truth_array = truth_ds.ReadAsArray() != -1

    prediction_ds = gdal.Open(prediction_file)
    prediction_array = prediction_ds.ReadAsArray()

    assert truth_array.shape == prediction_array.shape

    # 5 possible ways to split the data
    height, width = prediction_array.shape
    train_idx = (slice(None), slice(width // 2, None))  # right
    val_idx = (slice(height // 2, None), slice(None, width // 2))  # bottom left
    test_idx = (slice(None, height // 2), slice(None, width // 2))  # top left
    val_test_idx = (slice(None), slice(None, width // 2))  # left
    all_idx = (slice(None), slice(None))  # all

    # Calculate statistics for each subset
    print('Train:')
    calculate_statistics(truth_array[train_idx], prediction_array[train_idx])
    print('Validation:')
    calculate_statistics(truth_array[val_idx], prediction_array[val_idx])
    print('Test:')
    calculate_statistics(truth_array[test_idx], prediction_array[test_idx])
    print('Validation + Test:')
    calculate_statistics(
        truth_array[val_test_idx], prediction_array[val_test_idx])
    print('All:')
    calculate_statistics(truth_array[all_idx], prediction_array[all_idx])


def stacked_metrics(truth_file, dirname):
    """Combine all of the predictions for the time-series into a single
    segmentation according to majority vote.

    Parameters:
        truth_file (str): file containing the ground truth labels
        dirname (str): directory containing the predictions
    """
    truth_ds = gdal.Open(truth_file)
    truth_array = truth_ds.ReadAsArray() != -1

    prediction_array = np.zeros(truth_array.shape)
    for j in range(12):
        filename = os.path.join(
            dirname, 'all_predictions_{}.tif'.format(j))
        prediction_ds = gdal.Open(filename)
        prediction_array += prediction_ds.ReadAsArray()

    prediction_array /= 12
    prediction_array = prediction_array > 0.5

    assert truth_array.shape == prediction_array.shape

    # 5 possible ways to split the data
    height, width = prediction_array.shape
    train_idx = (slice(None), slice(width // 2, None))  # right
    val_idx = (slice(height // 2, None), slice(None, width // 2))  # bottom left
    test_idx = (slice(None, height // 2), slice(None, width // 2))  # top left
    val_test_idx = (slice(None), slice(None, width // 2))  # left
    all_idx = (slice(None), slice(None))  # all

    # Calculate statistics for each subset
    print('Train:')
    calculate_statistics(truth_array[train_idx], prediction_array[train_idx])
    print('Validation:')
    calculate_statistics(truth_array[val_idx], prediction_array[val_idx])
    print('Test:')
    calculate_statistics(truth_array[test_idx], prediction_array[test_idx])
    print('Validation + Test:')
    calculate_statistics(
        truth_array[val_test_idx], prediction_array[val_test_idx])
    print('All:')
    calculate_statistics(truth_array[all_idx], prediction_array[all_idx])


if __name__ == '__main__':
    # Parse supplied arguments
    parser = set_up_parser()
    args = parser.parse_args()

    for i in range(4):
        print('\nLevel', i)
        dirname = os.path.join(args.prediction_dir, 'l{}'.format(i))

        if i == 0:
            filename = os.path.join(dirname, 'test_predictions.tif')
            compute_subsets(args.ground_truth, filename)
        elif i == 1:
            stacked_metrics(args.ground_truth, dirname)
        else:
            filename = os.path.join(dirname, 'all_predictions.tif')
            compute_subsets(args.ground_truth, filename)
