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

class Gauss(nn.Module):
    def __init__(self, args):
        super(Gauss, self).__init__()
        self.args = args
        self.sigma = nn.Parameter(torch.randn((1, ), dtype=torch.float))
        self.mu = nn.Parameter(torch.randn((1, ), dtype=torch.float))

    def forward(self, x):
        return torch.sigmoid(self.mu + self.sigma * torch.randn_like(x, dtype=torch.float, device='cuda' if args.cuda else 'cpu')).mean(dim=1)


class Constant(nn.Module):
    def __init__(self, args):
        super(Constant, self).__init__()
        self.args = args
        self.constant = nn.Parameter(0.1 * torch.randn(1, dtype=torch.float))

    def forward(self, x):
        return torch.sigmoid(self.constant * torch.ones((x.shape[0], x.shape[2], x.shape[3]), dtype=torch.float, device='cuda' if self.args.cuda else 'cpu'))
