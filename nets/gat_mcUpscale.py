"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
from dgl.nn.pytorch.conv.gatconv import GATConv
import dgl
import sys
import torch.nn.functional as F


class GATMC(nn.Module):
    def __init__(self, g, gnn_layers, in_dim, num_hidden, grid_width, image_width, heads, activation, feat_drop,
                 attn_drop, negative_slope, residual, cnn_layers):
        super(GATMC, self).__init__()
        # print('hidden', num_hidden[-1], 'heads', heads,  'imgw', image_width, 'chann')
        # assert (heads[-1] == 1), 'The number of output heads must be the number of expected channels'
        assert (len(heads) == gnn_layers), 'The number of elements in the heads list must be the number of layers'
        assert (len(num_hidden) == gnn_layers), 'The number of elements in the num_hidden list must be the number of layers'
        self.num_channels = num_hidden[-1]

        self.first = True
        self.g = g
        self.num_hidden = num_hidden
        self.grid_width = grid_width
        self.image_width = image_width
        self.gnn_layers = gnn_layers
        self.cnn_layers = cnn_layers
        self.layers = nn.ModuleList()
        self.activation = activation
        print('Heads {}'.format(heads))
        # input projection (no residual)

        print(in_dim, num_hidden[0], heads[0], '(0)')
        self.layers.append(GATConv(
            in_dim, num_hidden[0], heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))

        # hidden layers
        for l in range(1, gnn_layers-1):
            print(num_hidden[l-1] * heads[l-1], num_hidden[l], heads[l], '('+str(l)+')')
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.layers.append(GATConv(
                num_hidden[l-1] * heads[l-1], num_hidden[l], heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))

        print(num_hidden[-1] * heads[-1], num_hidden[-1], heads[-1], '(*)')
        self.layers.append(GATConv(
            num_hidden[-2] * heads[-2], num_hidden[-1], heads[-1],
            feat_drop, attn_drop, negative_slope, residual, torch.sigmoid)) #

        print('CNN LAYERS', self.cnn_layers)
        if self.cnn_layers == 1:
            self.conv1 = nn.ConvTranspose2d(in_channels=self.num_channels, out_channels=1, kernel_size=7, stride=3, padding=2)
        elif self.cnn_layers == 2:
            self.conv1 = nn.ConvTranspose2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=5, stride=2, padding=1)
            self.bnorm1 = nn.BatchNorm2d(self.num_channels)
            self.conv3 = nn.ConvTranspose2d(in_channels=self.num_channels, out_channels=1, kernel_size=3, stride=2, padding=1)
        elif self.cnn_layers == 3:
            self.conv1 = nn.ConvTranspose2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=3, padding=1)
            self.bnorm1 = nn.BatchNorm2d(self.num_channels)
            self.conv2 = nn.ConvTranspose2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=3, padding=1)
            self.bnorm2 = nn.BatchNorm2d(self.num_channels)
            self.conv3 = nn.ConvTranspose2d(in_channels=self.num_channels, out_channels=1, kernel_size=3, padding=1)

    def set_g(self, g):
        self.g = g

    def forward(self, inputs):
        
        h = inputs
        if self.first:
            print('H: ', h.size())
        for l in range(self.gnn_layers-1):
            h = self.layers[l](self.g, h).permute(0,2,1).flatten(1)
            if self.first:
                print('out {}: {}'.format(l, h.size()))
        # output projection
        if self.first:
            print(h.shape)
        logits = self.layers[-1](self.g, h)
        logits = logits.permute(0,2,1).mean(2) #maybe without this
        if self.first:
            print('OUT {}'.format(logits.size()))
        # return logits
        # return logits[getMaskForBatch(self.g)]

        # print logits[getMaskForBatch(self.g)]
        self.first = False

        base_index = 0
        batch_number = 0
        unbatched = dgl.unbatch(self.g)
        # print('graphs', len(unbatched), self.image_width)
        output = torch.Tensor(size=(len(unbatched), self.image_width, self.image_width))
        for g in unbatched:
            num_nodes = g.number_of_nodes()
            
            first = base_index
            last =  base_index + self.grid_width*self.grid_width - 1
            # print(self.grid_width, last-first)
            x1_prev = logits[first:last+1, :].view(1, self.grid_width, self.grid_width, self.num_channels)
            x1_permute = x1_prev.permute(0, 3, 1, 2).flatten(1)
            x1 = x1_permute.view(1, self.num_channels, self.grid_width, self.grid_width)
            if self.cnn_layers == 1:
                x4 = torch.sigmoid(self.conv1(x1))
            elif self.cnn_layers == 2:
                x3 = self.bnorm1(F.relu(self.conv1(x1)))
                x4 = torch.sigmoid(self.conv3(x3))
            elif self.cnn_layers == 3:
                x2 = self.bnorm1(F.relu(self.conv1(x1)))
                x3 = self.bnorm2(F.relu(self.conv2(x2)))
                x4 = torch.sigmoid(self.conv3(x3))

            output[batch_number, :, :] = x4.view(self.image_width, self.image_width)
            base_index += num_nodes
            batch_number += 1
        return output

    

