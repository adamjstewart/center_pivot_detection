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
from dataset import XRays
from scipy import misc
import imageio


# TODO: Write up CNN architecture.
class 
class Gauss(nn.Modules):
    def __init__(self, args):
        super(WhereAreTheCirclesCNN, self).__init__()
        self.args = args
        self.sigma = nn.Parameter(torch.rand((1, ), dtype=torch.float))
        self.mu = torch.Parameter(torch.randn((1, ), dtype=torch.float))

    def forward(self, x):
        return F.tanh(self.mu + self.sigma * torch.randn_like(x, device='cuda' if args.cuda else 'cpu'))


architectures = {
    0: Gauss,
}
##########
### Test
##########
def test(args, model, test_loader):
    model.eval()
    acc = []
    if args.cuda:
        model = model.cuda()
    for batch_idx, (x, y) in enumerate(test_loader):
        if args.cuda:
            x = x.cuda()
            y = y.cuda()
        yhat = model(x)

        ### Compute
        batch_acc = ((yhat > 0) == y).mean()
        print("acc[{}]={}".format(batch_idx, batch_acc))
        acc.append(batch_acc)
    print('Net accuracy: {}'.format(np.mean(acc)))


##########
### argparse
##########
parser = argparse.ArgumentParser()
parser.add_argument("--base_output", dest="base_output", default="outputs/screw_this/", help="Directory which will have folders per run")  # noqa
parser.add_argument("--run", dest='run_code', type=str, default='', help='Name this run. It will be part of file names for convenience')  # noqa
parser.add_argument("--tensorboard_dir", dest="tensorboard_dir", type=str, default="tensorboard", help="Subdirectory to save logs using tensorboard")  # noqa
parser.add_argument("--model_dir", dest="model_dir", type=str, default="models", help="Subdirectory to save models")  # noqa
parser.add_argument("--image_dir", dest="image_dir", type=str, default="images", help="Subdirectory to save images")  # noqa

# misc
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1337, help="Random seed (default=1337)")  # noqa
parser.add_argument("--cuda", dest="cuda", default=False, action="store_true")  # noqa
parser.add_argument("--debug", default=False, action="store_true", help="Debug mode")
parser.add_argument("--e", dest="n_epochs", default=100, type=int, help="Number of epochs")
parser.add_argument("--o", dest="output_dir", default="test_images_128x128", type=str, help="Folder to dump images in")
parser.add_argument("--lr", dest="lr", type=float, metavar='<float>', default=0.001, help='Learning rate')  # noqa
parser.add_argument("--weight_decay", dest="weight_decay", type=float, metavar='<float>', default=1e-5, help='Weight decay')  # noqa
parser.add_argument("--arch", dest="arch", type=str, metavar='<float>', default='rand', help='Architecture')  # noqa
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
model = architectures[args.arch](args)
if args.cuda:
    model = model.cuda()

adam = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
for epoch in range(args.n_epochs):
    losses = []
    for bidx, (x, y) in enumerate(train_loader):
        bs = x.shape[0]
        if args.cuda:
            x = x.cuda()
            y = y.cuda()
        yhat = model(x)
        loss = F.binary_cross_entropy(yhat.view(bs, -1), y.view(bs, -1))
        losses.append(loss.item())
        adam.zero_grad()
        loss.backward()
        adam.step()
    print("loss[{}]={}".format(epoch, np.mean(losses)))

##########
### Le end
##########
print("Used run_code: {}".format(args.run_code))
