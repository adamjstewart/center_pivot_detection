"""

"""

import os
import json
import argparse
import numpy as np
import scipy
import random
from random import shuffle
from time import time
import sys
import pdb
import pickle as pk
from collections import defaultdict
import itertools
# pytorch imports
import torch
from torch import autograd
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.sampler as samplers
from modules.unet import (
    UNet_cat,
    UNet_add,
)


class CNNTS3D_add(nn.Module):
    def __init__(self, args, loadable_state_dict=None, inC=6, inT=12, embedding_size=12):
        # assert(inT == 12, "Adam, you said I'd get 12 time steps. No more, no less.")
        super(CNNTS3D_add, self).__init__()
        self.args = args
        self.inC, self.inT = inC, inT
        self.embedding_size = embedding_size
        # Temporal Conv
        self.temporal_embedder = nn.Sequential(  # T = 12
            nn.Conv2d(inC, 16, (3, 1), stride=(1, 1), padding=(1, 0)),  # T = 12
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, (4, 1), stride=(2, 1), padding=(1, 0)),  # T = 6
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, embedding_size, (4, 1), stride=(2, 1), padding=(1, 0)),  # T = 3
            nn.LeakyReLU(negative_slope=0.2),
        )
        # Spatial
        self.unet_add = UNet_add(args, in_channels=3 * embedding_size)


        if loadable_state_dict:
            self.load_state_dict(loadable_state_dict)

    def forward(self, x):
        # x : bs * C * T * H * W
        # -> y: N * H * W
        N, C, T, H, W = (x.shape)
        x = x.view(N, C, T, H*W)
        x = self.temporal_embedder(x)  # N * embedding_size * 1 * HW
        x = x.view(N, self.embedding_size * 3, 1, H, W).view(N, -1, H, W)
        return self.unet_add(x)


class CNNTS3D_cat(nn.Module):
    def __init__(self, args, loadable_state_dict=None, inC=6, inT=12, embedding_size=16):
        # assert(inT == 12, "Adam, you said I'd get 12 time steps. No more, no less.")
        super(CNNTS3D_cat, self).__init__()
        self.args = args
        self.inC, self.inT = inC, inT
        self.embedding_size = embedding_size
        # Temporal Conv
        self.temporal_embedder = nn.Sequential(  # T = 12
            nn.Conv2d(inC, 16, (3, 1), stride=(1, 1), padding=(1, 0)),  # T = 12
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, (4, 1), stride=(2, 1), padding=(1, 0)),  # T = 5
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, embedding_size, (4, 1), stride=(2, 1), padding=(1, 0)),  # T = 3
            nn.LeakyReLU(negative_slope=0.2),
        )
        # Spatial
        self.unet_cat = UNet_cat(args, in_channels=3 * embedding_size)


        if loadable_state_dict:
            self.load_state_dict(loadable_state_dict)

    def forward(self, x):
        # x : bs * C * T * H * W
        # -> y: N * H * W
        N, C, T, H, W = (x.shape)
        x = x.view(N, C, T, H*W)
        x = self.temporal_embedder(x)  # N * embedding_size * 1 * HW
        x = x.view(N, self.embedding_size * 3, 1, H, W).view(N, -1, H, W)
        return self.unet_cat(x)



class CNNTS2D_add(nn.Module):
    def __init__(self, args, loadable_state_dict=None, inC=6, inT=12):
        super(CNNTS2D_add, self).__init__()
        self.args = args
        self.inC, self.inT = inC, inT
        self.unet_add = UNet_add(args, in_channels=inC * inT)
        if loadable_state_dict:
            self.load_state_dict(loadable_state_dict)

    def forward(self, x):
        # x: N * C * T * H * W
        # -> y: N * H * W
        x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])
        return self.unet_add(x)

class CNNTS2D_cat(nn.Module):
    def __init__(self, args, loadable_state_dict=None, inC=6, inT=12):
        super(CNNTS2D_cat, self).__init__()
        self.args = args
        self.inC, self.inT = inC, inT
        self.unet_cat = UNet_cat(args, in_channels=inC * inT)
        if loadable_state_dict:
            self.load_state_dict(loadable_state_dict)

    def forward(self, x):
        # x: N * C * T * H * W
        # -> y: N * H * W
        x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])
        return self.unet_cat(x)
