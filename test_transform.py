#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: jjzhou012
@contact: jjzhou012@163.com
@file: test.py
@time: 2022/8/19 16:02
@desc:     测试脚本
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
import torch_geometric.transforms as T

from model.model import *
from utils.parameter import get_args
from utils.tools import data_split, EarlyStopping, setup_seed
from transform.perturbation import *

# parameter
args = get_args()

# setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data path
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'TU')
print(path)

# test transform
transforms = T.Compose([
    # T.NormalizeFeatures()
    # FeatureMasking(prob=0.5, target='node')
    Node_Random_Dropping(prob=0.1)
])

# load dataset
dataset_raw = TUDataset(path, name=args.dataset, use_node_attr=True, use_edge_attr=True)
dataset_aug = TUDataset(path, name=args.dataset, use_node_attr=True, use_edge_attr=True, transform=transforms)

print('\n################# dataset information #########################')
print(dataset_raw)
print('Num. of graphs:    {}'.format(len(dataset_raw)))
print('Ave. num. of nodes:  raw: {}    aug: {}'.format(np.mean([g.x.size(0) for g in dataset_raw]), np.mean([g.x.size(0) for g in dataset_aug])))
print('Ave. num. of edges:  raw: {}    aug: {}'.format(np.mean([g.edge_index.size(1) for g in dataset_raw]), np.mean([g.edge_index.size(1) for g in dataset_aug])))
print('Num. of node features:  raw: {}   aug: {}'.format(dataset_raw.num_node_features, dataset_aug.num_node_features))
print('Num. of edge features:  raw: {}   aug: {}\n'.format(dataset_raw.num_edge_features, dataset_aug.num_edge_features))

print('\n################# ################### #########################')


id = 0
data_1 = dataset_raw[id]
data_2 = dataset_aug[id]

print(data_1)
# print(data_1.x)
print(data_1.edge_index)
print(data_2.edge_index)

