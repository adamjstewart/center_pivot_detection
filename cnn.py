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
import multiprocessing
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
from modules.smolnet import SmolNet
from modules.dumb import Gauss, Constant
from modules.unet import (
    UNet,
    UNet_too_big,
)


architectures = {
    'randn': Gauss,
    'unet': UNet,
    'const': Constant,
    'smol': SmolNet,
}
##########
### Test
##########
def test(args, model, test_loader, dataset=None, prefix='', vis_file=''):
    model.eval()
    acc = []
    if args.cuda:
        model = model.cuda()
    if vis_file != '':
        predictions = []
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
        if vis_file != '':
            predictions.extend([(yhat[idx, :, :].detach().cpu().numpy(), y[idx], x[idx]) for idx in range(data.shape[0])])

    print('{} net accuracy: {}'.format(prefix, np.mean(acc)))
    if vis_file != '':
        assert(dataset is not None)
        dataset.write(predictions, vis_file)

##########
### argparse
##########
parser = argparse.ArgumentParser()
parser.add_argument('--d', dest='data_dir', type=str, default='data/baby_data', help='Data directory.')
parser.add_argument("--base_output", dest="base_output", default="outputs/", help="Directory which will have folders per run")  # noqa
parser.add_argument("--run", dest='run_code', type=str, default='', help='Name this run. It will be part of file names for convenience')  # noqa
parser.add_argument("--model_dir", dest="model_dir", type=str, default="models", help="Subdirectory to save models")  # noqa
parser.add_argument("--image_dir", dest="image_dir", type=str, default="images", help="Subdirectory to save images")  # noqa

# misc
parser.add_argument("--split", dest="split", type=float, metavar='<float>', default=0.9, help='Train/test split')  # noqa
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1337, help="Random seed (default=1337)")  # noqa
parser.add_argument("--cuda", dest="cuda", default=False, action="store_true")  # noqa
parser.add_argument("--debug", default=False, action="store_true", help="Debug mode")
parser.add_argument("--b", dest="batch_size", default=32, type=int, help="Batch size")
parser.add_argument("--v", dest="val_rate", default=8, type=int, help="Validation rate")
parser.add_argument("--e", dest="n_epochs", default=1000, type=int, help="Number of epochs")
parser.add_argument("--lr", dest="lr", type=float, metavar='<float>', default=0.001, help='Learning rate')  # noqa
parser.add_argument("--weight_decay", dest="weight_decay", type=float, metavar='<float>', default=1e-5, help='Weight decay')  # noqa
parser.add_argument("--arch", dest="arch", type=str, metavar='<float>', choices=architectures.keys(), default='unet', help='Architecture')  # noqa
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
args.model_dir = os.path.join(args.data_dir, args.run_code, args.model_dir)  # noqa
args.image_dir = os.path.join(args.data_dir, args.run_code)  # noqa

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

num_workers = multiprocessing.cpu_count()
train_loader, test_loader = utils.binary_splitter(dataset, args.split, batch_size=args.batch_size, num_workers=num_workers)

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
    if epoch % args.val_rate == 0:
        test(args, model, test_loader, prefix='val ', dataset=dataset, vis_file=os.path.join(args.image_dir, 'val_predictions.tif'))

test(args, model, train_loader, prefix='train ', dataset=dataset, vis_file=os.path.join(args.image_dir, 'train_predictions.tif'))
test(args, model, test_loader, prefix='test ', dataset=dataset, vis_file=os.path.join(args.image_dir, 'test_predictions.tif'))
test(args, model, itertools.chain(train_loader, test_loader), prefix='all ', dataset=dataset, vis_file=os.path.join(args.data_dir, args.run_code, 'all_predictions.tif'))

##########
### Le end
##########

print("Used run_code: {}".format(args.run_code))
