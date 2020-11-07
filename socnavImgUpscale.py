"""
This script generates the dataset.
"""

import sys
import os
import json
from collections import namedtuple
import math

import torch as th
import cv2
import dgl
import numpy as np
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import save_info, load_info

grid_width = 18  # 30 #18
output_width = 73  # 121 #73
area_width = 800.  # Spatial area of the grid

limit = 32000  # 31191

path_saves = 'saves/'  # This variable is necessary due tu a bug in dgl.DGLDataset source code


#  human to wall distance
def dist_h_w(h, wall):
    hxpos = float(h['xPos']) / 100.
    hypos = float(h['yPos']) / 100.
    wxpos = float(wall.xpos) / 100.
    wypos = float(wall.ypos) / 100.
    return math.sqrt((hxpos - wxpos) * (hxpos - wxpos) + (hypos - wypos) * (hypos - wypos))


# Calculate the closet node in the grid to a given node by its coordinates
def closest_grid_node(grid_ids, w_a, w_i, x, y):
    c_x = int((x * (w_i / w_a) + (w_i / 2)))
    c_y = int((y * (w_i / w_a) + (w_i / 2)))
    if 0 <= c_x < grid_width and 0 <= c_y < grid_width:
        return grid_ids[c_x][c_y]
    return None


def closest_grid_nodes(grid_ids, w_a, w_i, r, x, y):
    c_x = int((x * (w_i / w_a) + (w_i / 2)))
    c_y = int((y * (w_i / w_a) + (w_i / 2)))
    cols, rows = (int(math.ceil(r * w_i / w_a)), int(math.ceil(r * w_i / w_a)))
    rangeC = list(range(-cols, cols + 1))
    rangeR = list(range(-rows, rows + 1))
    p_arr = [[c, r] for c in rangeC for r in rangeR]
    grid_nodes = []
    r_g = r * w_i / w_a
    for p in p_arr:
        if math.sqrt(p[0] * p[0] + p[1] * p[1]) <= r_g:
            if 0 <= (c_x + p[0]) < grid_width and 0 <= (c_y + p[1]) < grid_width:
                grid_nodes.append(grid_ids[c_x + p[0]][c_y + p[1]])

    return grid_nodes


# Given the coordinates of a node in the grid: returns 1 if it is inside the room, else return 0
def inside_room(x, y, room):
    count = 0
    for idx in range(len(room) - 1):
        x1_w = room[idx][0] - x
        y1_w = room[idx][1] - y
        x2_w = room[idx + 1][0] - x
        y2_w = room[idx + 1][1] - y
        if (x1_w < 0 and x2_w < 0) or (y1_w * y2_w >= 0):
            continue

        if x1_w == x2_w:
            count += 1
        else:
            rw = (y2_w - y1_w) / (x2_w - x1_w)  # Slope
            b = y1_w - (rw * x1_w)
            x_corte = -(b / rw)
            if x_corte > 0:
                count += 1

    if count % 2 != 0:
        return 1, count  # It is inside the room
    return 0, count  # It is outside the room


def get_node_descriptor_header():
    node_descriptor_header = ['is_human', 'is_object', 'is_room', 'is_wall', 'is_grid',
                              'hum_x_pos', 'hum_y_pos',
                              'hum_orient_sin', 'hum_orient_cos',
                              'obj_x_pos', 'obj_y_pos',
                              'obj_orient_sin', 'obj_orient_cos',
                              'num_hs_room', 'num_hs2_room',
                              'wall_x_pos', 'wall_y_pos',
                              'wall_orient_sin', 'wall_orient_cos',
                              'grid_x_pos', 'grid_y_pos']
    return node_descriptor_header


def get_features():
    node_types_one_hot = ['human', 'object', 'room', 'wall', 'grid']
    human_metric_features = ['hum_x_pos', 'hum_y_pos', 'hum_orientation_sin', 'hum_orientation_cos']
    object_metric_features = ['obj_x_pos', 'obj_y_pos', 'obj_orientation_sin', 'obj_orientation_cos']
    room_metric_features = ['room_humans', 'room_humans2']
    wall_metric_features = ['wall_x_pos', 'wall_y_pos', 'wall_orientation_sin', 'wall_orientation_cos']
    grid_metric_features = ['grid_x_pos', 'grid_y_pos']  # , 'flag_inside_room']  # , 'count']
    all_features = node_types_one_hot + human_metric_features + object_metric_features + room_metric_features + wall_metric_features + grid_metric_features
    n_features = len(all_features)

    return n_features, all_features


def get_relations():
    room_set = {'l_p', 'l_o', 'l_w', 'l_g', 'p_p', 'p_o', 'p_g', 'o_g', 'w_g'}
    grid_set = {'g_c', 'g_ri', 'g_le', 'g_u', 'g_d', 'g_uri', 'g_dri', 'g_ule', 'g_dle'}
    # ^
    # |_p = person             g_ri = grid right
    # |_w = wall               g_le = grid left
    # |_l = lounge             g_u = grid up
    # |_o = object             g_d = grid down
    # |_g = grid node
    self_edges_set = {'P', 'O', 'W', 'L'}

    for e in list(room_set):
        room_set.add(e[::-1])
    relations_class = room_set | grid_set | self_edges_set
    relations = sorted(list(relations_class))

    return relations


def generate_grid_graph_data():
    print('Initialising grid graph')
    # Define variables for edge types and relations
    grid_rels = get_relations()
    edge_types = []  # List to store the relation of each edge
    edge_norms = []  # List to store the norm of each edge

    # Grid properties
    connectivity = 8  # Connections of each node
    node_ids = np.zeros((grid_width, grid_width), dtype=int)  # Array to store the IDs of each node
    typeMap = dict()
    coordinates_gridGraph = dict()  # Dict to store the spatial coordinates of each node
    src_nodes = []  # List to store source nodes
    dst_nodes = []  # List to store destiny nodes

    # Feature dimensions
    n_features, all_features = get_features()

    # Compute the number of nodes and initialize feature vectors
    n_nodes = grid_width ** 2
    features_gridGraph = th.zeros(n_nodes, n_features)

    max_used_id = -1
    for y in range(grid_width):
        for x in range(grid_width):
            max_used_id += 1
            node_id = max_used_id
            node_ids[x][y] = node_id

            # Self edges
            src_nodes.append(node_id)
            dst_nodes.append(node_id)
            edge_types.append(grid_rels.index('g_c'))
            edge_norms.append([1.])

            if x < grid_width - 1:
                src_nodes.append(node_id)
                dst_nodes.append(node_id + 1)
                edge_types.append(grid_rels.index('g_ri'))
                edge_norms.append([1.])
                if connectivity == 8 and y > 0:
                    src_nodes.append(node_id)
                    dst_nodes.append(node_id - grid_width + 1)
                    edge_types.append(grid_rels.index('g_uri'))
                    edge_norms.append([1.])
            if x > 0:
                src_nodes.append(node_id)
                dst_nodes.append(node_id - 1)
                edge_types.append(grid_rels.index('g_le'))
                edge_norms.append([1.])
                if connectivity == 8 and y < grid_width - 1:
                    src_nodes.append(node_id)
                    dst_nodes.append(node_id + grid_width - 1)
                    edge_types.append(grid_rels.index('g_dle'))
                    edge_norms.append([1.])
            if y < grid_width - 1:
                src_nodes.append(node_id)
                dst_nodes.append(node_id + grid_width)
                edge_types.append(grid_rels.index('g_d'))
                edge_norms.append([1.])
                if connectivity == 8 and x < grid_width - 1:
                    src_nodes.append(node_id)
                    dst_nodes.append(node_id + grid_width + 1)
                    edge_types.append(grid_rels.index('g_dri'))
                    edge_norms.append([1.])
            if y > 0:
                src_nodes.append(node_id)
                dst_nodes.append(node_id - grid_width)
                edge_types.append(grid_rels.index('g_u'))
                edge_norms.append([1.])
                if connectivity == 8 and x > 0:
                    src_nodes.append(node_id)
                    dst_nodes.append(node_id - grid_width - 1)
                    edge_types.append(grid_rels.index('g_ule'))
                    edge_norms.append([1.])

            typeMap[node_id] = 'g'  # 'g' for 'grid'
            x_pos = (-area_width / 2. + (x + 0.5) * (area_width / grid_width))
            y_pos = (-area_width / 2. + (y + 0.5) * (area_width / grid_width))
            features_gridGraph[node_id, all_features.index('grid')] = 1
            features_gridGraph[node_id, all_features.index('grid_x_pos')] = 2. * x_pos / 1000
            features_gridGraph[node_id, all_features.index('grid_y_pos')] = -2. * y_pos / 1000

            coordinates_gridGraph[node_id] = [x_pos / 1000, y_pos / 1000]

    src_nodes = th.LongTensor(src_nodes)
    dst_nodes = th.LongTensor(dst_nodes)

    edge_types = th.LongTensor(edge_types)
    edge_norms = th.Tensor(edge_norms)

    return src_nodes, dst_nodes, n_nodes, features_gridGraph, edge_types, edge_norms, coordinates_gridGraph, typeMap, \
           node_ids


def generate_room_graph_data(data):
    # Define variables for edge types and relations
    rels = get_relations()
    edge_types = []  # List to store the relation of each edge
    edge_norms = []  # List to store the norm of each edge

    # Compute data for walls
    Wall = namedtuple('Wall', ['orientation', 'xpos', 'ypos'])
    walls = []
    for wall_index in range(len(data['room']) - 1):
        p1 = np.array(data['room'][wall_index + 0])
        p2 = np.array(data['room'][wall_index + 1])
        dist = np.linalg.norm(p1 - p2)
        iters = int(dist / 400) + 1
        if iters > 1:
            v = (p2 - p1) / iters
            for i in range(iters):
                pa = p1 + v * i
                pb = p1 + v * (i + 1)
                inc2 = pb - pa
                midsp = (pa + pb) / 2
                walls.append(Wall(math.atan2(inc2[0], inc2[1]), midsp[0], midsp[1]))
        else:
            inc = p2 - p1
            midp = (p2 + p1) / 2
            walls.append(Wall(math.atan2(inc[0], inc[1]), midp[0], midp[1]))

    # Compute the number of nodes
    #      room +  room walls      + humans               + objects
    n_nodes = 1 + len(walls) + len(data['humans']) + len(data['objects'])

    # Feature dimensions
    n_features, all_features = get_features()
    features = th.zeros(n_nodes, n_features)

    # Nodes variables
    typeMap = dict()
    position_by_id = dict()
    src_nodes = []  # List to store source nodes
    dst_nodes = []  # List to store destiny nodes

    # Room (Global node)
    room_id = 0
    max_used_id = 0
    typeMap[room_id] = 'l'  # 'l' for 'room' (lounge)
    position_by_id[0] = [0, 0]
    features[room_id, all_features.index('room')] = 1.
    features[room_id, all_features.index('room_humans')] = len(data['humans'])
    features[room_id, all_features.index('room_humans2')] = len(data['humans']) * len(data['humans'])

    # humans
    for h in data['humans']:
        src_nodes.append(h['id'])
        dst_nodes.append(room_id)
        edge_types.append(rels.index('p_l'))
        edge_norms.append([1. / len(data['humans'])])

        src_nodes.append(room_id)
        dst_nodes.append(h['id'])
        edge_types.append(rels.index('l_p'))
        edge_norms.append([1.])

        typeMap[h['id']] = 'p'  # 'p' for 'person'
        max_used_id = max(h['id'], max_used_id)
        xpos = float(h['xPos']) / 1000.
        ypos = float(h['yPos']) / 1000.

        position_by_id[h['id']] = [xpos, ypos]
        orientation = float(h['orientation']) / 180. * math.pi
        while orientation > math.pi: orientation -= 2. * math.pi
        while orientation < -math.pi: orientation += 2. * math.pi
        if orientation > math.pi:
            orientation -= math.pi
        elif orientation < -math.pi:
            orientation += math.pi

        features[h['id'], all_features.index('human')] = 1.
        features[h['id'], all_features.index('hum_orientation_sin')] = math.sin(orientation)
        features[h['id'], all_features.index('hum_orientation_cos')] = math.cos(orientation)
        features[h['id'], all_features.index('hum_x_pos')] = 2. * xpos
        features[h['id'], all_features.index('hum_y_pos')] = -2. * ypos

    # objects
    for o in data['objects']:
        src_nodes.append(o['id'])
        dst_nodes.append(room_id)
        edge_types.append(rels.index('o_l'))
        edge_norms.append([1. / len(data['objects'])])

        src_nodes.append(room_id)
        dst_nodes.append(o['id'])
        edge_types.append(rels.index('l_o'))
        edge_norms.append([1.])

        typeMap[o['id']] = 'o'  # 'o' for 'object'
        max_used_id = max(o['id'], max_used_id)
        xpos = float(o['xPos']) / 1000.
        ypos = float(o['yPos']) / 1000.

        position_by_id[o['id']] = [xpos, ypos]
        orientation = float(o['orientation']) / 180. * math.pi
        while orientation > math.pi: orientation -= 2. * math.pi
        while orientation < -math.pi: orientation += 2. * math.pi
        features[o['id'], all_features.index('object')] = 1
        features[o['id'], all_features.index('obj_orientation_sin')] = math.sin(orientation)
        features[o['id'], all_features.index('obj_orientation_cos')] = math.cos(orientation)
        features[o['id'], all_features.index('obj_x_pos')] = 2. * xpos
        features[o['id'], all_features.index('obj_y_pos')] = -2. * ypos

    # walls
    wids = dict()
    for wall in walls:
        max_used_id += 1
        wall_id = max_used_id
        wids[wall] = wall_id
        typeMap[wall_id] = 'w'  # 'w' for 'walls'

        src_nodes.append(wall_id)
        dst_nodes.append(room_id)
        edge_types.append(rels.index('w_l'))
        edge_norms.append([1. / len(walls)])

        src_nodes.append(room_id)
        dst_nodes.append(wall_id)
        edge_types.append(rels.index('l_w'))
        edge_norms.append([1.])

        position_by_id[wall_id] = [wall.xpos / 1000, wall.ypos / 1000]
        features[wall_id, all_features.index('wall')] = 1.
        features[wall_id, all_features.index('wall_orientation_sin')] = math.sin(wall.orientation)
        features[wall_id, all_features.index('wall_orientation_cos')] = math.cos(wall.orientation)
        features[wall_id, all_features.index('wall_x_pos')] = 2. * wall.xpos / 1000.
        features[wall_id, all_features.index('wall_y_pos')] = -2. * wall.ypos / 1000.

    # interactions
    for link in data['links']:
        typeLdir = typeMap[link[0]] + '_' + typeMap[link[1]]
        typeLinv = typeMap[link[1]] + '_' + typeMap[link[0]]

        src_nodes.append(link[0])
        dst_nodes.append(link[1])
        edge_types.append(rels.index(typeLdir))
        edge_norms.append([1.])

        src_nodes.append(link[1])
        dst_nodes.append(link[0])
        edge_types.append(rels.index(typeLinv))
        edge_norms.append([1.])

    # self edges
    for node_id in range(n_nodes):
        r_type = typeMap[node_id].upper()

        src_nodes.append(node_id)
        dst_nodes.append(node_id)
        edge_types.append(rels.index(r_type))
        edge_norms.append([1.])

    src_nodes = th.LongTensor(src_nodes)
    dst_nodes = th.LongTensor(dst_nodes)

    edge_types = th.LongTensor(edge_types)
    edge_norms = th.Tensor(edge_norms)

    return src_nodes, dst_nodes, n_nodes, features, edge_types, edge_norms, position_by_id, typeMap


class SocNavDataset(DGLDataset):
    # Generate grid graph data just once
    grid_src, grid_dst, n_nodes_grid, features_grid, edge_types_grid, edge_norms_grid, coordinates_grid, \
    typeMapGrid, grid_ids = generate_grid_graph_data()

    rels = get_relations()

    def __init__(self, path, mode='train', raw_dir='data/', init_line=-1, end_line=-1, loc_limit=limit,
                 force_reload=False, verbose=True, debug=False):
        if type(path) is str:
            self.path = raw_dir + path
        else:
            self.path = path
        self.mode = mode
        self.init_line = init_line
        self.end_line = end_line
        self.graphs = []
        self.labels = []
        self.data = dict()
        self.data['typemaps'] = []
        self.data['coordinates'] = []
        self.data['identifiers'] = []
        self.debug = debug
        self.limit = loc_limit

        # Define device. GPU if it is available
        self.device = 'cpu'

        if self.debug:
            self.limit = 1 + (0 if init_line == -1 else init_line)

        super(SocNavDataset, self).__init__("SocNav", raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)

    def get_dataset_name(self):
        graphs_path = 'graphs_' + self.mode + '_s_' + str(limit) + '_g_' + str(grid_width) + '_l_' + str(
            output_width) + '.bin'
        info_path = 'info_' + self.mode + '_s_' + str(limit) + '_g_' + str(grid_width) + '_l_' + str(
            output_width) + '.pkl'
        return graphs_path, info_path

    #################################################################
    # Implementation of abstract methods
    #################################################################

    def download(self):
        # No need to download any data
        pass

    def process(self):

        if type(self.path) is str and self.path.endswith('.json'):
            linen = -1
            for line in open(self.path).readlines():
                if linen % 1000 == 0:
                    print(linen)

                if linen + 1 >= self.limit:
                    break
                linen += 1
                if self.init_line >= 0 and linen < self.init_line:
                    continue
                if linen > self.end_line >= 0:
                    continue

                raw_data = json.loads(line)
                final_graph = self.generate_final_graph(raw_data)
                self.graphs.append(final_graph)
            self.labels = th.tensor(self.labels, dtype=th.float64)
        elif type(self.path) == list and type(self.path[0]) == str:
            raw_data = json.loads(self.path)
            final_graph = self.generate_final_graph(raw_data)
            self.graphs.append(final_graph)
            self.labels = th.tensor(self.labels, dtype=th.float64)
        else:
            final_graph = self.generate_final_graph(self.path)
            self.graphs.append(final_graph)
            self.labels = th.tensor(self.labels, dtype=th.float64)

    def generate_final_graph(self, raw_data):
        # Generate data for the room graph for the specific line and identifier
        room_src, room_dst, n_nodes_room, features_room, edge_types_room, edge_norms_room, \
        coordinates_room, typeMapRoom = generate_room_graph_data(raw_data)

        # Store typemaps, coordinates and descriptor header as additional info
        typeMapRoomShift = dict()
        coordinates_roomShift = dict()

        for key in typeMapRoom:
            typeMapRoomShift[key + len(self.typeMapGrid)] = typeMapRoom[key]
            coordinates_roomShift[key + len(self.coordinates_grid)] = coordinates_room[key]

        coordinates = {**self.coordinates_grid, **coordinates_roomShift}
        typemap = {**self.typeMapGrid, **typeMapRoomShift}

        self.data['typemaps'].append(typemap)
        self.data['coordinates'].append(coordinates)
        self.data['identifiers'].append(raw_data['identifier'])
        self.data['descriptor_header'] = get_node_descriptor_header()

        # Storing labels
        if self.mode != 'run':
            label = cv2.cvtColor(cv2.imread('labels/all/' + self.data['identifiers'][-1] + '.png'),
                                 cv2.COLOR_BGR2GRAY).astype(float) / 255
            label = cv2.resize(label, (output_width, output_width), interpolation=cv2.INTER_CUBIC)
            label = label.reshape((1, output_width, output_width))
        else:
            label = np.zeros((1, output_width, output_width))

        self.labels.append(label)

        # Adding offsets and new edges
        room_src = room_src + self.n_nodes_grid
        room_dst = room_dst + self.n_nodes_grid

        # Link each node in the room graph to the correspondent grid graph.
        for r_n_id in range(1, n_nodes_room):
            r_n_type = typeMapRoom[r_n_id]
            x, y = coordinates_room[r_n_id]
            closest_grid_nodes_id = closest_grid_nodes(self.grid_ids, area_width, grid_width, 25., x * 1000,
                                                               y * 1000)
            for g_id in closest_grid_nodes_id:
                room_src = th.cat([room_src, th.tensor([g_id], dtype=th.int32)], dim=0)
                room_dst = th.cat([room_dst, th.tensor([r_n_id + self.n_nodes_grid], dtype=th.int32)], dim=0)
                edge_types_room = th.cat([edge_types_room, th.LongTensor([self.rels.index('g_' + r_n_type)])], dim=0)
                edge_norms_room = th.cat([edge_norms_room, th.Tensor([[1.]])])

                room_src = th.cat([room_src, th.tensor([r_n_id + self.n_nodes_grid], dtype=th.int32)], dim=0)
                room_dst = th.cat([room_dst, th.tensor([g_id], dtype=th.int32)], dim=0)
                edge_types_room = th.cat([edge_types_room, th.LongTensor([self.rels.index(r_n_type + '_g')])], dim=0)
                edge_norms_room = th.cat([edge_norms_room, th.Tensor([[1.]])])

        try:
            # Create final graph with the grid and room data
            final_graph = dgl.graph((th.cat([self.grid_src, room_src], dim=0),
                                             th.cat([self.grid_dst, room_dst], dim=0)),
                                            num_nodes=self.n_nodes_grid + n_nodes_room,
                                            idtype=th.int32, device=self.device)
            final_graph.ndata['h'] = th.cat([self.features_grid, features_room], dim=0).to(self.device)
            edge_types = th.cat([self.edge_types_grid, edge_types_room], dim=0).to(self.device)
            edge_norms = th.cat([self.edge_norms_grid, edge_norms_room], dim=0).to(self.device)
            final_graph.edata.update({'rel_type': edge_types, 'norm': edge_norms})
            return(final_graph)
        except Exception:
            raise


    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)

    def save(self):
        if self.debug:
            return
        # Generate paths
        graphs_path, info_path = tuple((path_saves + x) for x in self.get_dataset_name())
        os.makedirs(os.path.dirname(path_saves), exist_ok=True)

        # Save graphs
        save_graphs(graphs_path, self.graphs, {'labels': self.labels})

        # Save additional info
        save_info(info_path, {'typemaps': self.data['typemaps'],
                              'coordinates': self.data['coordinates'],
                              'identifiers': self.data['identifiers'],
                              'descriptor_header': self.data['descriptor_header']})

    def load(self):
        # Generate paths
        graphs_path, info_path = tuple((path_saves + x) for x in self.get_dataset_name())

        # Load graphs
        self.graphs, label_dict = load_graphs(graphs_path)
        self.labels = label_dict['labels']

        # Load info
        self.data['typemaps'] = load_info(info_path)['typemaps']
        self.data['coordinates'] = load_info(info_path)['coordinates']
        self.data['descriptor_header'] = load_info(info_path)['descriptor_header']
        self.data['identifiers'] = load_info(info_path)['identifiers']

    def has_cache(self):
        # Generate paths
        graphs_path, info_path = tuple((path_saves + x) for x in self.get_dataset_name())
        if self.debug:
            return False
        return os.path.exists(graphs_path) and os.path.exists(info_path)
