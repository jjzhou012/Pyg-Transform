#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: jjzhou012
@contact: jjzhou012@163.com
@file: parameter.py
@time: 2022/8/20 22:05
@desc:
'''
import argparse




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='COX2')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--wandb', action='store_true', help='Track experiment')
    # args = parser.parse_args()
    return parser.parse_args()