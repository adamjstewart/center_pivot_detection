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

class SmolNet(nn.Module):
    def __init__(self, args, in_channels=6):
        super(SmolNet, self).__init__()
        self.args = args
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 1, 1),  # 16, 256, 256
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(16, 32, 3, 1, 1),  # 32, 256, 256
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(32, 16, 3, 1, 1),  # 16, 256, 256
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(16, 1, 3, 1, 1),  # 16, 256, 256
        )
    def forward(self, x):
        return torch.sigmoid(self.net(x).view(x.shape[0], x.shape[2], x.shape[3]))
