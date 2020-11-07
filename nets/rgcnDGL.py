import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl
from functools import partial

from dgl.nn.pytorch.conv.relgraphconv import RelGraphConv

class RGCN(nn.Module):
    def __init__(self, g, gnn_layers, cnn_layers, in_dim, hidden_dimensions, grid_width, image_width, num_rels, activation, feat_drop, num_bases=-1):
        super(RGCN, self).__init__()
        self.g = g
        self.in_dim = in_dim
        self.hidden_dimensions = hidden_dimensions
        self.num_channels = hidden_dimensions[-1]
        self.num_rels = num_rels
        self.feat_drop = feat_drop
        self.num_bases = num_bases
        self.activation = activation
        self.gnn_layers = gnn_layers
        self.cnn_layers = cnn_layers
        self.grid_width = grid_width
        self.image_width = image_width
        # create RGCN layers
        self.build_model()
        
    def set_g(self, g):
        self.g = g

    def build_model(self):
        self.layers = nn.ModuleList()
        # input to hidden
        i2h = self.build_input_layer()
        self.layers.append(i2h)
        # hidden to hidden
        for i in range(self.gnn_layers-2):
            h2h = self.build_hidden_layer(i)
            self.layers.append(h2h)
        # hidden to output
        h2o = self.build_output_layer()
        self.layers.append(h2o)

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


    def build_input_layer(self):
        print('Building an INPUT  layer of {}x{}'.format(self.in_dim, self.hidden_dimensions[0]))
        return RelGraphConv(self.in_dim, self.hidden_dimensions[0], self.num_rels, regularizer='basis',
                            dropout=self.feat_drop, num_bases=self.num_bases, activation=F.leaky_relu)

    def build_hidden_layer(self, i):
        print('Building an HIDDEN  layer of {}x{}'.format(self.hidden_dimensions[i], self.hidden_dimensions[i+1]))
        return RelGraphConv(self.hidden_dimensions[i], self.hidden_dimensions[i+1],  self.num_rels, regularizer='basis',
                            dropout=self.feat_drop, num_bases=self.num_bases, activation=F.leaky_relu)

    def build_output_layer(self):
        print('Building an OUTPUT  layer of {}x{}'.format(self.hidden_dimensions[-2], self.hidden_dimensions[-1]))
        return RelGraphConv(self.hidden_dimensions[-2], self.hidden_dimensions[-1], self.num_rels, regularizer='basis',
                            dropout=self.feat_drop, num_bases=self.num_bases, activation=torch.sigmoid)

    def forward(self, features, etypes):
        h = features
        self.g.edata['norm'] = self.g.edata['norm'].to(device=features.device)

        #print("H",h.size())
        # self.g.ndataz['h'] = features
        for layer in self.layers:
            h = layer(self.g, h, etypes)
            #print("Size:", h.size())
        logits = h


        base_index = 0
        batch_number = 0
        unbatched = dgl.unbatch(self.g)
        output = torch.Tensor(size=(len(unbatched), self.image_width, self.image_width))
        for g in unbatched:
            num_nodes = g.number_of_nodes()
            first = base_index
            last =  base_index + self.grid_width*self.grid_width - 1

            x1_prev = logits[first:last+1, :].view(1, self.grid_width, self.grid_width, self.num_channels)
            x1_permute = x1_prev.permute(0, 3, 1, 2).flatten(1)
            x1 = x1_permute.view(1, self.num_channels, self.grid_width, self.grid_width)
            if self.cnn_layers == 1:
                x4 = torch.sigmoid(self.conv1(x1))
            elif self.cnn_layers == 2:
                #x3 = self.bnorm1(F.relu(self.conv1(x1)))
                x3 = F.relu(self.conv1(x1))
                x4 = torch.sigmoid(self.conv3(x3))
            elif self.cnn_layers == 3:
                x2 = self.bnorm1(F.relu(self.conv1(x1)))
                x3 = self.bnorm2(F.relu(self.conv2(x2)))
                x4 = torch.sigmoid(self.conv3(x3))

            output[batch_number, :, :] = x4.view(self.image_width, self.image_width)
            base_index += num_nodes
            batch_number += 1
        return output

