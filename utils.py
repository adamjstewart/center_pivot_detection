import pdb
import os
import pickle
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import torch
from torch.utils.data import SubsetRandomSampler
from collections import defaultdict

def binary_splitter(dataset, frac, shuffle=True, batch_size=32, debug=False, num_workers=4, selected_indices=None, confounder_group_size=None):
    dataset_size = len(dataset)
    if debug:
        dataset_size = 100

    if selected_indices:
        indices = selected_indices
    else:
        indices = list(range(dataset_size))

    if confounder_group_size:
        # floor frac to nearest confounder group size
        frac = float(int(frac * len(dataset) / confounder_group_size) * confounder_group_size) / float(len(dataset))

    split = int(np.floor(frac * dataset_size))

    if shuffle and not confounder_group_size:
        np.random.shuffle(indices)

    train_indices, test_indices = indices[:split], indices[split:]

    if shuffle and confounder_group_size:
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)
    return train_loader, test_loader
