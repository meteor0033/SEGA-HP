import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import math
import dgl
import sympy
import scipy
import numpy as np
from torch import nn
from torch.nn import init
#注意这里导入的几个模型
from torch_geometric.nn import GCNConv,GINConv,SAGEConv,GATConv,PNAConv, GraphSAGE, ChebConv

"""
     High-pass_GCN
     Paper: High-pass Graph convolution network for Graph anomaly detection
"""

class ChebConvGAD_c(nn.Module):
    ## in_feats 特征点维度；h_feats：隐层维度；num_classes：节点分类数（nomal，anomaly）
    def __init__(self, in_feats, h_feats, num_classes,  k=2, batch=False):
        super(ChebConvGAD_c, self).__init__()
        #self.lambda_max = max_eigenvalue
        print("K.....:",k)
        torch.manual_seed(1234567)
        self.conv1 = ChebConv(h_feats,h_feats,k)
        self.conv2 = ChebConv(h_feats,h_feats,k)
        #self.conv3 = ChebConv(h_feats,h_feats,k)
        self.linear1 = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats, h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()


    def forward(self, in_feat, data):
        h = self.linear1(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
 
        h = self.conv1(h, data.edge_index)
        h = self.conv2(h, data.edge_index)
        #h = self.conv3(data.x, data.edge_index)
        h = self.linear3(h)
        h = self.act(h)
        h = self.linear4(h)
        return h

class ChebConvGAD_s(nn.Module):
    ## in_feats 特征点维度；h_feats：隐层维度；num_classes：节点分类数（nomal，anomaly）
    def __init__(self, in_feats, h_feats, num_classes,  k=2, batch=False):
        super(ChebConvGAD_s, self).__init__()
        #self.lambda_max = max_eigenvalue
        print("K.....:",k)
        torch.manual_seed(1234567)
        self.conv1 = ChebConv(h_feats,h_feats,k)
        self.conv2 = ChebConv(h_feats,h_feats,k)
        #self.conv3 = ChebConv(h_feats,h_feats,k)
        self.linear1 = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats, h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()


    def forward(self, in_feat, data):
        h = self.linear1(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
 
        h = self.conv1(in_feat, data.edge_index)
        h = self.conv2(in_feat, data.edge_index)
        #h = self.conv3(data.x, data.edge_index)
        h = self.linear3(h)
        h = self.act(h)
        h = self.linear4(h)
        return h

class ChebConvGAD_j(nn.Module):
    ## in_feats 特征点维度；h_feats：隐层维度；num_classes：节点分类数（nomal，anomaly）
    def __init__(self, in_feats, h_feats, num_classes,  k=2, batch=False):
        super(ChebConvGAD_j, self).__init__()
        #self.lambda_max = max_eigenvalue
        print("K.....:",k)
        torch.manual_seed(1234567)
        self.conv1 = ChebConv(h_feats,h_feats,k)
        self.conv2 = ChebConv(h_feats,h_feats,k)
        #self.conv3 = ChebConv(h_feats,h_feats,k)
        self.linear1 = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats, h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()


    def forward(self, in_feat, data):
        h = self.linear1(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
 
        h = self.conv1(in_feat, data.edge_index)
        h = self.conv2(in_feat, data.edge_index)
        #h = self.conv3(data.x, data.edge_index)
        h = self.linear3(h)
        h = self.act(h)
        h = self.linear4(h)
        return h
