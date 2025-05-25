import torch
import os
import json
import pickle as pk
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset

class Graph_dataset(Dataset):
    def __init__(self, dataset_path, json_path, representation):
        super().__init__()
        self.dataset_path = dataset_path
        self.graph_list_file = get_list_files(json_path, representation)

    def len(self):
        return len(self.graph_list_file)

    def get(self, idx):
        with open (os.path.join(self.dataset_path, self.graph_list_file[idx]), "rb") as graph_file:
            graph = pk.load(graph_file)
            adj_matrix_old = graph['adj']
            loop_adj = adj_matrix_old + sp.eye(adj_matrix_old.shape[0])
            node_features = torch.FloatTensor(graph['node_features'])
            adj_matrix = torch.FloatTensor(loop_adj)
            edge_index = torch.nonzero(adj_matrix, as_tuple=False).t().contiguous()
            labels = torch.LongTensor(graph['node_labels'].flatten())

            data = Data(x=node_features, edge_index=edge_index, y=labels)
            return data



def create_path(capture, representation, set_type, graph_name):
    if set_type == "train" or set_type == "val" or set_type == "test_benign":
        path = os.path.join(capture, representation, "full_benign", graph_name)
    elif set_type == "test_mixed":
        path = os.path.join(capture, representation, "mixed", graph_name)
    elif set_type == "test_malicious":
       path = os.path.join(capture, representation, "full_malicious", graph_name) 
    return path
    
    


def get_list_files(json_path, representation):
    file_paths = []
    # read json object
    with open(json_path,  'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    # iter over json to create the original data path
    set_type = os.path.splitext(os.path.basename(json_path))[0]
    for capture, file_list in json_data.items():
        for graph_name in file_list:
            # capture/representation/full_benign or full_malicious or mixed/graph_x.pkl
            file_path = create_path(capture, representation, set_type, graph_name)
            file_paths.append(file_path)
    file_paths.sort()
    return file_paths