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
from torch_geometric.utils import dropout_adj, degree, to_undirected, subgraph, negative_sampling
from torch_geometric.nn import knn_graph
from typing import List, Union
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform

# from .utils import negative_sampling


class Feature_Random_Shuffling(BaseTransform):
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
        self.num_feat = num_feat

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
        else:
            raise ValueError("'target' must be in {'node', 'edge'}")

        return data

    def __repr__(self):
        return '{}(target={}, seed={}, num_feat)'.format(self.__class__.__name__, self.target, self.seed, self.num_feat)


class Feature_Random_Masking(BaseTransform):
    '''
    Perturbation-based > Feature Perturbation > Feature Masking > Random Masking
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
            raise ValueError("'target' must be in {'node', 'edge'}")

        return data

    def __repr__(self):
        return '{}(prob={}, target={}, fill_value={})'.format(self.__class__.__name__, self.prob, self.target, self.fill_value)


# TODO: how to design weighted feature masking
class Feature_weighted_Masking(BaseTransform):
    '''
    Perturbation-based > Feature Perturbation > Feature Masking > Weighted Masking
    Feature: node or edge
    '''

    def __init__(self, weight='', target='node', fill_value=.0):
        '''
        Args:
            weight: weighted masking according to which metric, str
            target: 'node' or 'edge', str
            fill_value: mask with value, float
        '''
        self.weight = weight
        self.target = target
        self.fill_value = fill_value

    def __call__(self, data):
        if self.target == 'node':
            pass
        elif self.target == 'edge':
            pass
        else:
            raise ValueError("'target' must be in {'node', 'edge'}")

        return data

    def __repr__(self):
        pass
        # return '{}(prob={}, target={}, fill_value={})'.format(self.__class__.__name__, self.prob, self.target, self.fill_value)


class Edge_Random_Removing(BaseTransform):
    '''
    Perturbation-based > Structure Perturbation > Edge Perturbation > Edge Removing > Random Removing
    '''

    def __init__(self, prob=None):
        '''
        Args:
            prob: removing probability, float=[.0, 1.0]
        '''
        self.prob = prob

    def __call__(self, data):
        edge_index, edge_attr = dropout_adj(edge_index=data.edge_index, edge_attr=data.edge_attr, p=self.prob, num_nodes=data.x.size(0))

        if edge_index.size(1) == 0:
            return data

        data.edge_index = edge_index
        data.edge_attr = edge_attr

        return data

    def __repr__(self):
        return '{}(prob={})'.format(self.__class__.__name__, self.prob)


# TODO: how to design weighted edge removing
class Edge_Weight_Removing(BaseTransform):
    '''
    Perturbation-based > Structure Perturbation > Edge Perturbation > Edge Removing > Weight Removing
    '''

    def __init__(self, weight=''):
        '''
        Args:
            weight: weighted removing according to which metric, str
        '''
        self.weight = weight

    def __call__(self, data):
        pass

    def __repr__(self):
        pass
        # return '{}(prob={})'.format(self.__class__.__name__, self.prob)

# TODO: please check
class Edge_Random_Addition(BaseTransform):
    '''
    Perturbation-based > Structure Perturbation > Edge Perturbation > Edge Addition > Random Addition
    '''
    def __init__(self, prob=None):
        '''
        Args:
            prob: addition probability, float=[.0, 1.0]   or int
        '''
        self.prob = prob
    def __call__(self, data):
        num_nodes = data.x.size(0)
        num_neg_samples = int(((num_nodes ** 2 - num_nodes) / 2 - data.edge_index.size(1)) * self.prob) if isinstance(self.prob, float) else self.prob
        neg_edges_index = negative_sampling(edge_index=data.edge_index, num_nodes=num_nodes, num_neg_samples=num_neg_samples)

        data.edge_index = torch.cat((data.edge_index, neg_edges_index), dim=1)

        if data.edge_attr is not None:
            new_edge_attr =  torch.zeros((num_neg_samples, data.edge_attr.size(1)))
            data.edge_attr = torch.cat((data.edge_attr, new_edge_attr), dim=0)

        return data

    def __repr__(self):
        return '{}(prob={})'.format(self.__class__.__name__, self.prob)



class Node_Random_Dropping(BaseTransform):
    '''
    Perturbation-based > Structure Perturbation > Node Perturbation > Node Dropping > Random Dropping
    '''
    def __init__(self, prob=None):
        '''
        Args:
            prob: node dropping probability, float = [.0, 1.]
        '''
        self.prob = prob

    def __call__(self, data):

        keep_mask = torch.empty(size=(data.x.size(0),), dtype=torch.float32).uniform_(0, 1) > self.prob
        # keep_mask[0] = True

        if keep_mask.sum().item() < 2:
            return data

        edge_index, edge_attr = subgraph(keep_mask, data.edge_index, data.edge_attr, relabel_nodes=True, num_nodes=data.x.size(0))

        subset = keep_mask.nonzero().squeeze()
        x = data.x[subset, :]

        data.x = x
        data.edge_index = edge_index
        data.edge_attr = edge_attr

        return data

    def __repr__(self):
        return '{}(prob={})'.format(self.__class__.__name__, self.prob)

