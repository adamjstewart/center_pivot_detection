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


class UNet_fc_pool(nn.Module):
    def __init__(self, args, loadable_state_dict=None, inC=6, inT=12):
        super(UNet_fc_pool, self).__init__()
        self.args = args
        self.inC, self.inT = inC, inT
        self.unet_cat = UNet_cat(args, in_channels=inC, do_sigmoid=False)
        self.fc_pool = nn.Sequential(
            nn.Linear(inT, 1),
            nn.Sigmoid(),
        )

        if loadable_state_dict:
            self.load_state_dict(loadable_state_dict)

    def forward(self, x):
        # x : bs * C * T * H * W
        # -> y: N * H * W
        N, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
        x = self.unet_cat(x).view(N, T, H, W)  # N * T, H, W
        x = x.permute(0, 2, 3, 1).contiguous().view(N * H * W, T)  # N, H, W, T
        x = self.fc_pool(x).view(N * H * W)  # N * H * W
        return x.view(N, H, W)
