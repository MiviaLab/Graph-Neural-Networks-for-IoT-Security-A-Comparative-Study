import torch
import os
import json
import pickle as pk
import scipy.sparse as sp
import numpy as np
from numpy import load
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import random


class Graph_dataset(Dataset):
    def __init__(self, dataset_path, json_path, representation, normalize, min_max=None):
        super().__init__()
        self.dataset_path = dataset_path
        if 'tdg_graph' == representation:
            self._tdg = True
            self.representation = "etdg_graph"
        else:
            self._tdg = False
            self.representation = representation
        if min_max == None:
            min_max = os.path.join(dataset_path, "IoT23_min_max")
        self.min, self.max = find_min_max_range(min_max, representation)
        self.graph_list_file = get_list_files(json_path, self.representation)
        self.normalize = normalize

    def len(self):
        return len(self.graph_list_file)

    def get(self, idx):
        with open(os.path.join(self.dataset_path, self.graph_list_file[idx]), "rb") as graph_file:
            # print("Graph ", self.graph_list_file[idx])
            graph = pk.load(graph_file)
            adj_matrix_old = graph['adj']
            loop_adj = adj_matrix_old + sp.eye(adj_matrix_old.shape[0])
            adj_norm = normalize_adj(loop_adj).toarray()
            # node_features = torch.FloatTensor(graph['node_features'])
            if self.normalize:
                node_features = torch.FloatTensor(normalize_attrs(
                    graph['node_features'], self.min, self.max, self._tdg))
                # assert torch.max(
                #     node_features) <= 1.0, f"max {torch.max(node_features)}: {os.path.join(self.dataset_path, self.graph_list_file[idx])}"
                # assert torch.count_nonzero(
                #     node_features < 0.0) == 0, f"input < 0.0: {torch.max(node_features)}: {os.path.join(self.dataset_path, self.graph_list_file[idx])}"
                # assert torch.count_nonzero(torch.isnan(
                #     node_features)) == 0, f"Input nan: {torch.max(node_features)}: {os.path.join(self.dataset_path, self.graph_list_file[idx])}"
            else:
                print("no-norm")
                node_features = torch.FloatTensor(graph['node_features'])

            adj_matrix = torch.FloatTensor(adj_norm)
            edge_index = torch.nonzero(
                adj_matrix, as_tuple=False).t().contiguous()
            # print("edge index:", edge_index)
            labels = torch.FloatTensor(graph['node_labels'].flatten())
            data = Data(x=node_features, edge_index=edge_index, y=labels)
            return data


def create_path(capture, representation, set_type, graph_name):
    if set_type == "train" or set_type == "val" or set_type == "test_benign":
        path = os.path.join(capture, representation, "full_benign", graph_name)
    elif set_type == "test_mixed":
        path = os.path.join(capture, representation, "mixed", graph_name)
    elif set_type == "test_malicious":
        path = os.path.join(capture, representation,
                            "full_malicious", graph_name)
    return path


def get_list_files(json_path, representation):
    file_paths = []
    # read json object
    with open(json_path,  'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    # iter over json to create the original data path
    set_type = os.path.splitext(os.path.basename(json_path))[0]
    for capture, file_lists in json_data.items():
        for graph_name_list in file_lists:
            for graph_name in graph_name_list:
                # capture/representation/full_benign or full_malicious or mixed/graph_x.pkl
                file_path = create_path(
                    capture, representation, set_type, graph_name)
                file_paths.append(file_path)
    # file_paths.sort()
    # random.shuffle(file_paths)
    return file_paths


def find_min_max_range(dataset_path, representation):
    with load(os.path.join(dataset_path,  "min_max_" + representation, 'min.npz')) as file:
        min = file['arr_0']
    with load(os.path.join(dataset_path, "min_max_" + representation, 'max.npz')) as file:
        max = file['arr_0']

    return min, max


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def normalize_attrs(nodes_features, min_values, max_values, tdg):
    if tdg:
        nodes_features = nodes_features[:, :57]
        min_values = min_values[:57]
        max_values = max_values[:57]
    # print("PRE node features: ", nodes_features)
    nodes_features = (nodes_features - min_values) / (max_values - min_values)
    # print("POST node features: ", nodes_features)
    nodes_features = np.nan_to_num(nodes_features, nan=0.0, posinf=0.0)
    assert torch.sum(torch.isnan(torch.FloatTensor(nodes_features))
                     == True) == 0, "features contains nan"

    return nodes_features
