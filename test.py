#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: jjzhou012
@contact: jjzhou012@163.com
@file: test.py
@time: 2022/8/19 16:02
@desc:
'''


import argparse
import numpy as np
import os.path as osp

import torch
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import MLP, GINConv, global_add_pool

from model.model import *
from utils.parameter import get_args
from utils.tools import data_split, EarlyStopping, setup_seed

# parameter
args = get_args()

# setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
init_wandb(name=f'GIN-{args.dataset_aug}', batch_size=args.batch_size, lr=args.lr,
           epochs=args.epochs, hidden_channels=args.hidden_channels,
           num_layers=args.num_layers, device=device)

# load dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'TU')
print(path)
dataset = TUDataset(path, name=args.dataset_aug).shuffle()


train_splits, val_splits, test_splits = data_split(X=np.arange(len(dataset)),
                                                   Y=np.array([dataset[i].y.item() for i in range(len(dataset))]),  # 等官方修复bug
                                                   # Y=np.array([dataset[i].y[0].item() for i in range(len(dataset))]),
                                                   seeds=args.seeds[:args.exp_num], K=args.k_ford)


train_dataset = dataset[len(dataset) // 10:]
train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)

test_dataset = dataset[:len(dataset) // 10]
test_loader = DataLoader(test_dataset, args.batch_size)

# model
model = Net(dataset.num_features, args.hidden_channels, dataset.num_classes,
            args.num_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# training and testing
def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.batch).argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset_aug)


for epoch in range(1, args.epochs + 1):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    log(Epoch=epoch, Loss=loss, Train=train_acc, Test=test_acc)