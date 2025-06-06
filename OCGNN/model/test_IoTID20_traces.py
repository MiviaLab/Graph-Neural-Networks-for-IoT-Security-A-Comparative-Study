import torch
import numpy as np
import os
import csv
import time
import argparse
import pickle as pk
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from OCGNN import OCGNN
from torch_geometric.nn import GraphSAGE


class Graph_dataset(Dataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = dataset_path
        self.graph_list_file = os.listdir(dataset_path)

    def len(self):
        return len(self.graph_list_file)

    def get(self, idx):
        with open(os.path.join(self.dataset_path, self.graph_list_file[idx]), "rb") as graph_file:
            graph = pk.load(graph_file)
            adj_matrix_old = graph['adj']
            loop_adj = adj_matrix_old + sp.eye(adj_matrix_old.shape[0])
            adj_norm = normalize_adj(loop_adj).toarray()
            node_features = torch.FloatTensor(graph['node_features'])
            adj_matrix = torch.FloatTensor(adj_norm)
            edge_index = torch.nonzero(
                adj_matrix, as_tuple=False).t().contiguous()
            labels = torch.FloatTensor(graph['node_labels']).flatten()
            data = Data(x=node_features, edge_index=edge_index, y=labels)
            data.generate_ids()
            return data


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def get_best_model(args, directory_path):
    """
    Accesses the specified directory and selects the file with the highest numerical index
    in the file name. Assumes that the files have a common prefix followed by a numerical index.

    Parameters
    ----------
    directory_path (str): The path of the directory containing the model files.

    Returns
    ----------
    str: The name of the file with the highest numerical index, or None if the directory is empty or contains no valid files.
    """
    # List to store the numerical indices of the files
    model_indices = []
    try:
        # List all files in the directory
        files = os.listdir(directory_path)
        # Filter files that start with the prefix "OCGNN_model_"
        model_files = [f for f in files if f.startswith("OCGNN_model_")]
        # Extract numerical indices from the file names
        for model_file in model_files:
            try:
                index = int(model_file.split("_")[-1])
                model_indices.append((model_file, index))
            except ValueError:
                # Ignore files that do not have a valid numerical index
                continue
        # If there are no valid files, return None
        if not model_indices:
            return None
        # Sort the list of tuples based on the numerical index and select the file with the highest index
        best_model_file = sorted(
            model_indices, key=lambda x: x[1], reverse=True)[0][0]

        if args.graph_type == "similarity_graph" or "trajectory_graph":
            in_dim = 57
        if args.graph_type == "etdg_graph":
            in_dim = 72

        model = OCGNN(in_dim=in_dim,
                      hid_dim=args.hid_dim,
                      num_layers=args.num_layers,
                      dropout=args.dropout,
                      beta=args.beta,
                      backbone=GraphSAGE)

        model.load_state_dict(torch.load(
            os.path.join(directory_path, best_model_file)))
        print("Extracting ", best_model_file, "...")

        return model

    except FileNotFoundError:
        print(f"The directory {directory_path} does not exist.")
        return None


def compute_evaluation_metrics(threshold, y_scores, y_true):
    """
    Compute evaluation metrics for binary classification tasks.

    Parameters:
    - threshold (float): The decision threshold for classifying anomalies. 
                         Anomaly scores greater than this value are classified as anomalous (1), 
                         and scores less than or equal to this value are classified as normal (0).

    - y_scores (list of float): An array of anomaly scores, typically between 0 and 1, 
                                where higher values indicate higher likelihood of being an anomaly.

    - y_true (list of int): An array of true labels where 1 indicates an anomalous instance 
                            and 0 indicates a normal instance.

    Returns:
    - accuracy (float): The proportion of correctly classified instances.
    - precision (float): The proportion of true positive instances among the instances that are 
                         predicted as positive.
    - recall (float): The proportion of true positive instances among the instances that are 
                      actually positive.
    - f_score (float): The weighted harmonic mean of precision and recall.

    Note:
    The function assumes that the input arrays y_scores and y_true have the same length.
    """
    y_pred = [1. if score > threshold else 0. for score in y_scores]
    if len(y_pred) != len(y_true):
        print("Error")
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    try:
        for i in range(len(y_pred)):
            if y_pred[i] == 1 and y_true[i] == 1:
                tp += 1
            if y_pred[i] == 0 and y_true[i] == 0:
                tn += 1
            if y_pred[i] == 1 and y_true[i] == 0:
                fp += 1
            if y_pred[i] == 0 and y_true[i] == 1:
                fn += 1

        print("\nTP:", tp)
        print("\nTN:", tn)
        print("\nFP:", fp)
        print("\nFN:", fn)

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f_score = 2 * (precision * recall) / (precision + recall)
    except:
        precision = 0
        recall = 0
        f_score = 0

    return accuracy, precision, recall, f_score


def evaluate(args, model):
    """
    This function evaluates the model on the dataset selected and writes the results to a CSV file.

    Parameters
    ----------
    args: Command-line arguments or a configuration object containing parameters like dataset folder, graph type, etc.
    model: The model to be evaluated.

    Returns
    ----------
    None. Writes evaluation metrics to a CSV file.
    """
    # paths definition
    if args.graph_type == "etdg_graph":
        if args.dataset == "IoTID20":
            dataset_folder = os.path.join(
                args.dataset_path, "IoTID20_graph_rapresentation/extended_graph")
        if args.dataset == "IoT_traces":
            dataset_folder = os.path.join(
                args.dataset_path, "IoT_traces_graph_rapresentation/extended_graph")
    else:
        if args.dataset == "IoTID20":
            dataset_folder = os.path.join(
                args.dataset_path, "IoTID20_graph_rapresentation", args.graph_type)
        if args.dataset == "IoT_traces":
            dataset_folder = os.path.join(
                args.dataset_path, "IoT_traces_graph_rapresentation", args.graph_type)

    # Dataloaders definition
    graph_set = Graph_dataset(dataset_folder)
    test_dataloader = DataLoader(graph_set, num_workers=0)

    # check if GPU is available
    if args.device == 'cuda':
        device = torch.device(args.device)

    center = torch.load(os.path.join(
        args.center_path, args.graph_type, 'center.pt')).to(args.device)
    radius = torch.load(os.path.join(
        args.radius_path, args.graph_type, 'radius.pt')).to(args.device)

    total_time = 0.0
    score_list = []
    labels_list = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            emb = model(x, edge_index)
            start_time = time.time()  # seconds
            _, anomaly_score, _ = model.loss_func(emb, center, radius)
            end_time = time.time()

            score_list.extend(anomaly_score.numpy().tolist())
            labels_list.extend(data.y.numpy().tolist())
            elapsed_time = end_time - start_time
            total_time += elapsed_time

    average_pred_time = total_time / len(test_dataloader)

    opt_threshold = 0.
    accuracy, precision, recall, fscore = compute_evaluation_metrics(
        opt_threshold, score_list, labels_list)
    return accuracy, precision, recall, fscore, average_pred_time


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--hid_dim',
                        type=int,
                        default=28,
                        help='Dimension of hidden embedding (default: 28)')

    parser.add_argument('--num_layers',
                        type=int,
                        default=2,
                        help='Number of layers')

    parser.add_argument('--lr',
                        type=float,
                        default=3e-4,
                        help='Learning rate')

    parser.add_argument('--beta',
                        type=float,
                        default=0.1,
                        help='Hyperparam to control the trade-off between the sphere volume and the penalties')

    parser.add_argument('--eps',
                        type=float,
                        default=0.01,
                        help='Slack parameter (float)')

    parser.add_argument('--weight_decay',
                        type=float,
                        default=5e-4,
                        help='Weight for L2 loss')

    parser.add_argument('--dropout',
                        type=float,
                        default=0.5,
                        help='Dropout rate')

    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        help='GPU = cuda/CPU = cpu')

    parser.add_argument("--graph_type",
                        type=str,
                        default="trajectory_graph",
                        help="Graph type to test")

    parser.add_argument("--checkpoint_path",
                        type=str,
                        default="/user/sranieri/Thesis/Results/OCGNN/checkpoints",
                        help="Folder from which take the model to evaluate")

    parser.add_argument("--dataset_path",
                        type=str,
                        default="/user/sranieri/Thesis/Dataset/IoT23",
                        help="Path dataset to evaluate")

    parser.add_argument("--dataset",
                        type=str,
                        default="IoT_traces")

    parser.add_argument("--json_folder",
                        type=str,
                        default="/user/sranieri/BORSA/Dataset",
                        help="Dataset folder in json format from which take the dataset split")

    parser.add_argument("--result_path",
                        type=str,
                        default="/user/sranieri/Thesis/pvalue/OCGNN/test_mixed/y_pred_true",
                        help="Folder where to save test results")

    parser.add_argument("--center_path",
                        type=str,
                        default="/user/sranieri/Thesis/Results/OCGNN/center",
                        help="Folder where to read center value")

    parser.add_argument("--radius_path",
                        type=str,
                        default="/user/sranieri/Thesis/Results/OCGNN/radius",
                        help="Folder where to save radius value")

    parser.add_argument("--normalize",
                        type=int,
                        default=1)

    parser.add_argument("--debug",
                        action='store_true')

    parser.add_argument("--wandb_log",
                        action='store_true')

    args = parser.parse_args()

    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger")
        debugpy.wait_for_client()

    result_folder = os.path.join(args.result_path, args.dataset)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    csv_results_folder = os.path.join(result_folder, args.graph_type)
    # create csv results folder if it does not exists
    if not os.path.exists(csv_results_folder):
        os.makedirs(csv_results_folder)
    if args.dataset == "IoTID20":
        test_result_file = open(os.path.join(
            csv_results_folder, 'IoTID20_result.csv'), "w", newline='')
        test_writer = csv.writer(test_result_file)
        test_writer.writerow(
            ['Set', 'Accuracy', 'Precision',
             'Recall', 'F-Score', "TP", "TN", "FP", "FN", 'Average pred time(ms)'])

        model = get_best_model(args, os.path.join(
            args.checkpoint_path, args.graph_type))

        print("\n Starting the evaluation for IOTID20, {}...".format(args.graph_type))
        accuracy, precision, recall, fscore, average_pred_time = evaluate(
            args, model)
        test_writer.writerow(
            [accuracy, precision, recall, fscore, average_pred_time*1000])
        test_result_file.close()

    if args.dataset == "IoT_traces":
        test_result_file = open(os.path.join(
            csv_results_folder, 'IoT_traces_result.csv'), "w", newline='')
        test_writer = csv.writer(test_result_file)
        test_writer.writerow(
            ['Set', 'Accuracy', 'Precision',
             'Recall', 'F-Score', "TP", "TN", "FP", "FN", 'Average pred time(ms)'])

        model = get_best_model(args, os.path.join(
            args.checkpoint_path, args.graph_type))

        print("\n Starting the evaluation for IOT_traces, {}...".format(
            args.graph_type))
        accuracy, precision, recall, fscore, average_pred_time = evaluate(
            args, model)
        test_writer.writerow(
            [accuracy, precision, recall, fscore, average_pred_time*1000])
        test_result_file.close()
