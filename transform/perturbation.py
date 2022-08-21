#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: jjzhou012
@contact: jjzhou012@163.com
@file: perturbation_based.py
@time: 2022/8/20 22:26
@desc:
'''
import torch
from torch.distributions.bernoulli import Bernoulli
import numpy as np
import random
from torch_sparse import SparseTensor
from torch_geometric.utils import dropout_adj, degree, to_undirected, subgraph
from torch_geometric.nn import knn_graph
from typing import List, Union
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


class FeatureMasking(BaseTransform):
    '''
    Perturbation-based > Feature Perturbation > Feature Masking
    Feature: node or edge
    '''
    def __init__(self, prob=.0, target='node', fill_value=.0):
        '''
        Args:
            prob: masking probability, float=[.0, 1.0]
            target: 'node' or 'edge', str
            fill_value: mask with value, float
        '''
        self.prob = prob
        self.target = target
        self.fill_value = fill_value

    def __call__(self, data):
        if self.target == 'node':
            drop_mask = torch.empty(size=(data.x.size(1),), dtype=torch.float32).uniform_() < self.prob
            data.x[:, drop_mask] = self.fill_value
        elif self.target == 'edge':
            drop_mask = torch.empty(size=(data.edge_attr.size(1),), dtype=torch.float32).uniform_() < self.prob
            data.edge_attr[:, drop_mask] = self.fill_value
        else:
            raise

        return data

    def __repr__(self):
        return '{}(prob={}, target={}, fill_value={})'.format(self.__class__.__name__, self.prob, self.target, self.fill_value)


class FeatureShuffling(BaseTransform):
    '''
    Perturbation-based > Feature Perturbation > Feature Shuffling
    Feature: node or edge
    '''
    def __init__(self, target='node', seed=None, num_feat=None):
        '''
        Args:
            target: 'node' or 'edge', str
            seed: random seed, int
            num_feat：dim of attribute,
        '''
        self.target = target
        self.seed = seed
        self.num_feat=num_feat

    def __call__(self, data):
        if self.target == 'node':
            if self.num_feat:
                shuffle_idx = torch.randperm(n=self.num_feat)
                data.x[:, :self.num_feat] = data.x[:, :self.num_feat][:, shuffle_idx]
            else:
                shuffle_idx = torch.randperm(n=data.x.size(1))
                data.x = data.x[:, shuffle_idx]
        elif self.target == 'edge':
            if self.num_feat:
                shuffle_idx = torch.randperm(n=self.num_feat)
                data.edge_attr[:, :self.num_feat] = data.edge_attr[:, :self.num_feat][:, shuffle_idx]
            else:
                shuffle_idx = torch.randperm(n=data.edge_attr.size(1))
                data.edge_attr = data.edge_attr[:, shuffle_idx]

        return data

    def __repr__(self):
        return '{}(target={}, seed={}, num_feat)'.format(self.__class__.__name__, self.target, self.seed, self.num_feat)




