import torch
import pickle as pk
import os
import argparse
from torch_geometric.loader import DataLoader
from OCGNN import OCGNN
from torch_geometric.nn import GraphSAGE
from Graph_dataset import Graph_dataset
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")


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


def compute_y_pred(threshold, y_scores):

    y_pred = [1. if score > threshold else 0. for score in y_scores]

    return y_pred


def evaluate_IoT23(args, model, dataset):

    # paths definition
    json_set = os.path.join(args.json_folder, f"{dataset}.json")

    # Dataloaders definition
    # dataset_path, json_path, representation, normalize
    graph_set = Graph_dataset(dataset_path=args.dataset_path,
                              json_path=json_set,
                              representation=args.graph_type,
                              normalize=args.normalize)
    test_dataloader = DataLoader(graph_set, num_workers=0)

    # check if GPU is available
    if args.device == 'cuda':
        device = torch.device(args.device)

    center = torch.load(os.path.join(
        args.center_path, args.graph_type, 'center.pt')).to(args.device)
    radius = torch.load(os.path.join(
        args.radius_path, args.graph_type, 'radius.pt')).to(args.device)
    score_list = []
    labels_list = []
    model.to(device)
    model.eval()
    print("Testing {} set...".format(dataset))
    with torch.no_grad():
        for data in tqdm(test_dataloader):
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            emb = model(x, edge_index)
            _, anomaly_score, _ = model.loss_func(emb, center, radius)
            score_list.extend(anomaly_score.numpy().tolist())
            labels_list.extend(data.y.numpy().tolist())

    opt_threshold = 0.
    y_pred = compute_y_pred(opt_threshold, score_list)
    dest_file = os.path.join(
        args.result_path, args.graph_type, f"OCGNN_test_preds_{dataset}.pkl")
    with open(dest_file, 'wb') as file:
        pk.dump((labels_list, y_pred), file)


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

    final_path = os.path.join(args.result_path, args.graph_type)
    if not os.path.exists(final_path):
        os.makedirs(final_path)

    model = get_best_model(args, os.path.join(
        args.checkpoint_path, args.graph_type))

    evaluate_IoT23(args, model, "val")
    evaluate_IoT23(args, model, "test_benign")
    evaluate_IoT23(args, model, "test_malicious")
    evaluate_IoT23(args, model, "test_mixed")
