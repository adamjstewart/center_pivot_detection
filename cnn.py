"""
CNN for circle detection

python cnn.py --help
(we have argparse)
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

from datasets import landsat
from transforms import transforms as custom_transforms
import torchvision

from scipy import misc
import imageio
import utils

# TODO: Write up CNN architecture.
class SmolUNet(nn.Module):
    def __init__(self, args, in_channels=6):
        super(SmolUNet, self).__init__()
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


class UNet(nn.Module):
    def __init__(self, args, in_channels=6):
        super(UNet, self).__init__()
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


architectures = {
    'randn': Gauss,
    'unet': UNet,
    'const': Constant,
    'smol': SmolUNet,
}
##########
### Test
##########
def test(args, model, test_loader, prefix=''):
    model.eval()
    acc = []
    if args.cuda:
        model = model.cuda()
    for batch_idx, (data, target, y, x) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
            target = target.cuda()
        target = (target > 0).float()
        yhat = model(data)
        ### Compute
        batch_acc = ((yhat.detach() > 0.5) == (target > 0)).float().mean().item()
        # print("acc[{}]={}".format(batch_idx, batch_acc))
        acc.append(batch_acc)
    print('{} net accuracy: {}'.format(prefix, np.mean(acc)))


##########
### argparse
##########
parser = argparse.ArgumentParser()
parser.add_argument('--d', dest='data_dir', type=str, default='data/baby_data', help='Data directory.')
parser.add_argument("--base_output", dest="base_output", default="outputs/screw_this/", help="Directory which will have folders per run")  # noqa
parser.add_argument("--run", dest='run_code', type=str, default='', help='Name this run. It will be part of file names for convenience')  # noqa
parser.add_argument("--model_dir", dest="model_dir", type=str, default="models", help="Subdirectory to save models")  # noqa
parser.add_argument("--image_dir", dest="image_dir", type=str, default="images", help="Subdirectory to save images")  # noqa

# misc
parser.add_argument("--split", dest="split", type=float, metavar='<float>', default=0.9, help='Train/test split')  # noqa
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1337, help="Random seed (default=1337)")  # noqa
parser.add_argument("--cuda", dest="cuda", default=False, action="store_true")  # noqa
parser.add_argument("--debug", default=False, action="store_true", help="Debug mode")
parser.add_argument("--e", dest="n_epochs", default=10, type=int, help="Number of epochs")
parser.add_argument("--lr", dest="lr", type=float, metavar='<float>', default=0.01, help='Learning rate')  # noqa
parser.add_argument("--weight_decay", dest="weight_decay", type=float, metavar='<float>', default=1e-5, help='Weight decay')  # noqa
parser.add_argument("--arch", dest="arch", type=str, metavar='<float>', choices=architectures.keys(), default='randn', help='Architecture')  # noqa
args = parser.parse_args()


args.cuda = args.cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
os.makedirs(args.base_output, exist_ok=True)
if len(args.run_code) == 0:
    # Generate a run code by counting number of directories in oututs
    run_count = len(os.listdir(args.base_output))
    args.run_code = 'run{}'.format(run_count)
print("Using run_code: {}".format(args.run_code))
# If directory doesn't exist, create it
args.model_dir = os.path.join(args.base_output, args.run_code, args.model_dir)  # noqa
args.image_dir = os.path.join(args.base_output, args.run_code, args.image_dir)  # noqa

directories_needed = [args.model_dir, args.image_dir]

for dir_name in directories_needed:
    os.makedirs(dir_name, exist_ok=True)


##########
### Train
##########

# dataset stuff

baby_data_normalization_transform = torchvision.transforms.Normalize(
    mean=[
        519.2332344309812,
        800.9046442974101,
        789.4069866067395,
        3089.939039506722,
        2606.206511476599,
        1661.5216229707723
    ],
    std=[
        1110.5749716985256,
        1140.5262261747455,
        1164.0975479971798,
        1445.48526409245,
        1469.7028140666473,
        1396.344680066366
    ]
)
baby_data_transform = torchvision.transforms.Compose([
    custom_transforms.ToTensor(),
    baby_data_normalization_transform,
])

dataset = landsat.SingleScene(
    root=args.data_dir,
    size=256,
    transform=baby_data_transform
)

train_loader, test_loader = utils.binary_splitter(dataset, args.split)

model = architectures[args.arch](args)
if args.cuda:
    model = model.cuda()

adam = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

for epoch in range(args.n_epochs):
    losses = []
    accs = []
    # plusses = []
    for bidx, (data, target, y, x) in enumerate(train_loader):
        bs = x.shape[0]
        if args.cuda:
            data = data.cuda()
            target = target.cuda()

        target = (target > 0).float()
        pred = model(data)

        batch_acc = ((pred.detach() > 0.5) == (target > 0)).float().mean().item()

        accs.append(batch_acc)
        # plusses.append((target > 0).float().mean().item())

        loss = F.binary_cross_entropy(pred.view(bs, -1), target.view(bs, -1))
        losses.append(loss.item())
        adam.zero_grad()
        loss.backward()
        adam.step()
    print("loss[{}]={}".format(epoch, np.mean(losses)))
    print("acc[{}]={}".format(epoch, np.mean(accs)))
    # print("max_acc[{}]={}".format(epoch, np.mean(plusses)))

test(args, model, train_loader, prefix='train ')
test(args, model, test_loader, prefix='test ')

##########
### Le end
##########
print("Used run_code: {}".format(args.run_code))
