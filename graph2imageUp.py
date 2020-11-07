from torch.utils.data import DataLoader
import cv2
import dgl
import torch
import numpy as np
import sys
import socnavImgUpscale as socnavImg
import pickle
import torch.nn.functional as F

sys.path.append('nets')

#from gat import GAT
from gat_mcUpscale import GATMC
from rgcnDGL import RGCN

if torch.cuda.is_available() is True:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def collate(batch):
    graphs = [batch[0][0]]
    labels = batch[0][1]
    for graph, label in batch[1:]:
        graphs.append(graph)
        labels = torch.cat([labels, label], dim=0)
    batched_graphs = dgl.batch(graphs).to(torch.device(device))
    labels.to(torch.device(device))

    return batched_graphs, labels


class SNGNN2D(object):
    def __init__(self, base, device='cpu'):
        super(SNGNN2D, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device2 = torch.device(device)
        self.params = pickle.load(open(base+'/SNGNN2D.prms', 'rb'), fix_imports=True)
        # print(self.params)
        if self.params['net'] in [ 'gat' ]:
            self.GNNmodel = GAT(g=None,
                                    gnn_layers=self.params['gnn_layers'],
                                    in_dim=self.params['num_feats'],
                                    num_hidden=self.params['num_hidden'],  # feats hidden
                                    image_width=self.params['image_width'],
                                    heads=self.params['heads'],
                                    activation=self.params['F'],  # F.relu?
                                    feat_drop=self.params['in_drop'],
                                    attn_drop=self.params['attn_drop'],
                                    negative_slope=self.params['alpha'],
                                    residual=self.params['residual'],
                                    cnn_layers=self.params['cnn_layers'])

        elif self.params['net'] in [ 'gatmc' ]:
            self.GNNmodel = GATMC(g=None,
                                    gnn_layers=self.params['gnn_layers'],
                                    in_dim=self.params['num_feats'],
                                    num_hidden=self.params['num_hidden'],  # feats hidden
                                    grid_width=self.params['grid_width'],
                                    image_width=self.params['image_width'],
                                    heads=self.params['heads'],
                                    activation=self.params['F'],  # F.relu?
                                    feat_drop=self.params['in_drop'],
                                    attn_drop=self.params['attn_drop'],
                                    negative_slope=self.params['alpha'],
                                    residual=self.params['residual'],
                                    cnn_layers=self.params['cnn_layers'])
        elif self.params['net'] in [ 'rgcn' ]:
            self.GNNmodel = RGCN(g=None,
                                    gnn_layers=self.params['gnn_layers'],
                                    cnn_layers=self.params['cnn_layers'],
                                    in_dim=self.params['num_feats'],
                                    hidden_dimensions=self.params['num_hidden'],  # feats hidden
                                    grid_width=self.params['grid_width'],
                                    image_width=self.params['image_width'],
                                    num_rels=30,
                                    activation=self.params['F'],  # F.relu?
                                    feat_drop=self.params['in_drop'],
                                    num_bases=self.params['num_rels'])
        else:
            print('Unknown/unsupported model in the parameters file')
            sys.exit(0)


        self.GNNmodel.load_state_dict(torch.load(base+'/SNGNN2D.tch', map_location = device))
        self.GNNmodel.to(self.device)
        self.GNNmodel.eval()
        torch.set_grad_enabled(False)

    def predict(self, file, line):
        net_type = 'gat'
        graph_type = 'dani3'
        # print('We are in SNGNN2D.predict and the scenario provided is {}'.format(type(scenario)))
        test_dataset = socnavImg.SocNavDataset(file, mode = 'run', init_line=line, debug=True)
        test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=collate)

        for batch, data in enumerate(test_dataloader):
            subgraph, labels = data
            feats = subgraph.ndata['h']
            feats = feats.to(self.device)
            self.GNNmodel.set_g(subgraph)
            if self.params['net'] in [ 'rgcn' ]:
                logits = self.GNNmodel(feats.float(), subgraph.edata['rel_type'].squeeze().to(self.device))
            else:
                    logits = self.GNNmodel(feats.float())
            # print('l', logits.shape)
            # logits = (logits[socnavImg.getMaskForBatch(subgraph)].detach().to(self.device2).numpy())*255.
            logits = (logits[0].detach().to(self.device2).numpy())
            #print('ppp', logits.shape)


        # print ('In predict() a: ', np.min(logits), np.max(logits))
        _, logits = cv2.threshold(logits, 1., 1., cv2.THRESH_TRUNC)
        _, logits = cv2.threshold(logits, 0., 0., cv2.THRESH_TOZERO)
        # print ('In predict() b: ', np.min(logits), np.max(logits))

        return logits


class Human(object):
    def __init__(self, id, xPos, yPos, angle):
        super(Human, self).__init__()
        self.id = id
        self.xPos = xPos
        self.yPos = yPos
        self.angle = angle


class Object(object):
    def __init__(self, id, xPos, yPos, angle):
        super(Object, self).__init__()
        self.id = id
        self.xPos = xPos
        self.yPos = yPos
        self.angle = angle


class SNScenario(dict):
    def __init__(self):
        super(SNScenario, self).__init__()
        self.room = []
        self.humans = []
        self.objects = []
        self.interactions = []
        self.rebuild_structure()

    def add_room(self, sn_room):
        self.room = sn_room
        self.rebuild_structure()

    def add_human(self, sn_human):
        self.humans.append(sn_human)
        self.rebuild_structure()

    def add_object(self, sn_object):
        self.objects.append(sn_object)
        self.rebuild_structure()

    def add_interaction(self, sn_interactions):
        self.interactions.append(sn_interactions)
        self.rebuild_structure()

    def rebuild_structure(self):
        # Adding Robot
        self['identifier'] = "no identifier"
        self['robot'] = {'id': 0}
        # Adding Room
        self['room'] = self.room
        # Adding humans and objects
        self['humans'] = []
        self['objects'] = []
        for _human in self.humans:
            human = {}
            human['id'] = int(_human.id)
            human['xPos'] = float(_human.xPos)
            human['yPos'] = float(_human.yPos)
            human['orientation'] = float(_human.angle)
            self['humans'].append(human)
        for object in self.objects:
            Object = {}
            Object['id'] = int(object.id)
            Object['xPos'] = float(object.xPos)
            Object['yPos'] = float(object.yPos)
            Object['orientation'] = float(object.angle)
            self['objects'].append(Object)
        # Adding links
        self['links'] = []
        for interaction in self.interactions:
            link = []
            link.append(int(interaction[0]))
            link.append(int(interaction[1]))
            link.append('interact')
            self['links'].append(link)
        self['score'] = 0
