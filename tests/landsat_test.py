"""Tests the Landsat Dataset."""

from datasets import SingleScene
from torch.utils import data

import numpy as np
import pytest


def test_write():
    """Test ability to write prediction labels."""

    dataset = SingleScene()
    dataloader = data.DataLoader(dataset)

    predictions = []
    for _, target, y, x in dataloader:
        predictions.append((target, y, x))

    dataset.write(predictions, 'tests/test.tif')
