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


class UNet(nn.Module):
    def __init__(self, args, in_channels=6):
        super(UNet, self).__init__()
        self.args = args

        # c,256x256 -> 256x256

        # Level 0
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 1, 1),  # 16, 256, 256
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),  # 16, 256, 256
        )
        self.maxpool0 = nn.MaxPool2d(2, return_indices=True)  # 16, 128, 128
        self.deconv0 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),  # 16, 256, 256
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),  # 16, 256, 256
            nn.LeakyReLU(),
            nn.Conv2d(16, 1, 3, 1, 1),  # 1, 256, 256
        )
        self.maxunpool0 = nn.MaxUnpool2d(2)

        # Level 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),  # 32, 128, 128
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),  # 32, 128, 128
            nn.LeakyReLU(),
        )
        self.deconv1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),  # 32, 128, 128
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, 3, 1, 1),  # 16, 128, 128
            nn.LeakyReLU(),
        )
        self.maxpool1 = nn.MaxPool2d(2, return_indices=True)
        self.maxunpool1 = nn.MaxUnpool2d(2)

        # Level 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),  # 64, 64, 64
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),  # 64, 64, 64
            nn.LeakyReLU(),
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.LeakyReLU(),
        )
        self.maxpool2 = nn.MaxPool2d(2, return_indices=True)
        self.maxunpool2 = nn.MaxUnpool2d(2)

        # Level 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),  # 128, 64, 64
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, 3, 1, 1),  # 64, 64, 64
            nn.LeakyReLU(),
        )

    def forward(self, x):
        # x: bs, nbands, 256, 256
        x0 = x
        x1 = self.conv0(x0)
        x1p5, ind0 = self.maxpool0(x1)
        x2 = self.conv1(x1p5)
        x2p5, ind1 = self.maxpool1(x2)
        x3 = self.conv2(x2p5)
        x3p5, ind2 = self.maxpool2(x3)
        x4 = self.conv3(x3p5)
        # x4p5, ind3 = self.maxpool3(x4)
        # y4 = self.conv4(x4p5)  # 16, 16
        # Now, deconv like mad.
        # y3p5 = self.maxunpool3(y4, ind3)  # 32, 32
        # y3 = self.deconv3(torch.cat([x4, y3p5], dim=1))
        y2p5 = self.maxunpool2(x4, ind2)  # 64, 64
        y2 = self.deconv2(torch.cat([x3, y2p5], dim=1))
        y1p5 = self.maxunpool1(y2, ind1)  # 128, 128
        y1 = self.deconv1(torch.cat([x2, y1p5], dim=1))
        y0p5 = self.maxunpool0(y1, ind0)
        y0 = self.deconv0(torch.cat([x1, y0p5], dim=1))
        return torch.sigmoid(y0.view(x.shape[0], x.shape[2], x.shape[3]))



class UNet_too_big(nn.Module):
    def __init__(self, args, in_channels=6):
        super(UNet_too_big, self).__init__()
        self.args = args

        # c,256x256 -> 256x256

        # Level 0
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),  # 64, 256, 256
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),  # 64, 256, 256
        )
        self.maxpool0 = nn.MaxPool2d(2, return_indices=True)  # 64, 128, 128
        self.deconv0 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),  # 64, 256, 256
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),  # 64, 256, 256
            nn.LeakyReLU(),
            nn.Conv2d(64, 1, 3, 1, 1),  # 1, 256, 256
        )
        self.maxunpool0 = nn.MaxUnpool2d(2)

        # Level 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),  # 128, 128, 128
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),  # 128, 128, 128
            nn.LeakyReLU(),
        )
        self.deconv1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),  # 128, 128, 128
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, 3, 1, 1),  # 128, 128, 128
            nn.LeakyReLU(),
        )
        self.maxpool1 = nn.MaxPool2d(2, return_indices=True)
        self.maxunpool1 = nn.MaxUnpool2d(2)

        # Level 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),  # 256, 64, 64
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),  # 256, 64, 64
            nn.LeakyReLU(),
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.LeakyReLU(),
        )
        self.maxpool2 = nn.MaxPool2d(2, return_indices=True)
        self.maxunpool2 = nn.MaxUnpool2d(2)

        # Level 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),  # 512, 64, 64
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),  # 512, 64, 64
            nn.LeakyReLU(),
        )
        self.deconv3 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),  # 512, 32, 32
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.LeakyReLU(),
        )
        self.maxpool3 = nn.MaxPool2d(2, return_indices=True)
        self.maxunpool3 = nn.MaxUnpool2d(2)

        # Level 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1, 1),  # 512, 16, 16
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        # x: bs, nbands, 256, 256
        x0 = x
        x1 = self.conv0(x0)
        x1p5, ind0 = self.maxpool0(x1)
        x2 = self.conv1(x1p5)
        x2p5, ind1 = self.maxpool1(x2)
        x3 = self.conv2(x2p5)
        x3p5, ind2 = self.maxpool2(x3)
        x4 = self.conv3(x3p5)
        x4p5, ind3 = self.maxpool3(x4)
        y4 = self.conv4(x4p5)  # 16, 16
        # Now, deconv like mad.
        y3p5 = self.maxunpool3(y4, ind3)  # 32, 32
        y3 = self.deconv3(torch.cat([x4, y3p5], dim=1))
        y2p5 = self.maxunpool2(y3, ind2)  # 64, 64
        y2 = self.deconv2(torch.cat([x3, y2p5], dim=1))
        y1p5 = self.maxunpool1(y2, ind1)  # 128, 128
        y1 = self.deconv1(torch.cat([x2, y1p5], dim=1))
        y0p5 = self.maxunpool0(y1, ind0)
        y0 = self.deconv0(torch.cat([x1, y0p5], dim=1))
        return torch.sigmoid(y0.view(x.shape[0], x.shape[2], x.shape[3]))
