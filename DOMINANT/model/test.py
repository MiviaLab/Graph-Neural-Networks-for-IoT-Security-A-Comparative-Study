import torch
import os
import csv
import time
import argparse
import numpy as np
import random
from torch_geometric.loader import DataLoader
from DOMINANT import DOMINANT
from Graph_dataset import Graph_dataset
from functional import objective_function
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm
import warnings
from collections import OrderedDict
warnings.filterwarnings("ignore")

def seed_everything(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        # Filter files that start with the prefix "DOMINANT_model_"
        model_files = [f for f in files if f.startswith("DOMINANT_model_")]
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

        if args.graph_type == "tdg_graph":
            in_dim = 57
        if args.graph_type == "etdg_graph":
            in_dim = 72#15#72

        model = DOMINANT(in_dim=in_dim,
                         hid_dim=args.hidden_dim,
                         encoder_layers=args.encoder_layers,
                         decoder_layers=args.decoder_layers,
                         dropout=args.dropout)

        model.load_state_dict(torch.load(
            os.path.join(directory_path, best_model_file), map_location=torch.device('cpu')))
        print("Extracting ", best_model_file, "...")

        return model

    except FileNotFoundError:
        print(f"The directory {directory_path} does not exist.")
        return None


def import_optimal_threshold(threshold_folder):
    """
    Reads the content of the file named "optimal_threshold.txt" found in the specified directory.

    Parameters
    ----------
    directory_path : str
        The path of the directory from which to fetch the file.

    Returns
    ----------
    float
        The content of the file as a float, or None if the file is not found or an error occurs.
    """

    try:
        # Build the full path of the file named "optimal_threshold.txt"
        file_path = os.path.join(threshold_folder, "optimal_threshold.txt")

        # Check if the file exists
        if not os.path.isfile(file_path):
            print(
                f"The file optimal_threshold.txt does not exist in the directory {threshold_folder}.")
            return None

        # Open and read the file
        with open(file_path, 'r') as file:
            threshold = file.read()

        return float(threshold)

    except FileNotFoundError:
        print(
            f"The directory {threshold_folder} or the file optimal_threshold.txt does not exist.")
        return None
    except IOError:
        print(
            f"An error occurred while reading the file optimal_threshold.txt in {threshold_folder}.")
        return None
    except ValueError:
        print(
            f"The content of optimal_threshold.txt in {threshold_folder} could not be converted to float.")
        return None

def find_fp(y_true, y_pred):
    fp = 0
    for i in range(len(y_true)):
        if y_true[i] == 0 and y_pred[i] == 1:
            fp += 1
    return fp


def find_optimal_threshold(y_true, y_scores, threshold):
    threshold_list = []
    fp_list = []
    y_pred = [1.0 if score > threshold else 0.0 for score in y_scores]
    fp = find_fp(y_true, y_pred)
    print("Number of fp:", fp)
    fp_percentage = (fp / len(y_pred)) * 100
    print("Starting fp percentage:", fp_percentage)
    threshold_list.append(threshold)
    fp_list.append(fp)
    final_threshold = threshold
    while fp_percentage > 1:
        final_threshold += final_threshold * 0.05
        print(final_threshold)
        y_pred = [1.0 if score > final_threshold else 0.0 for score in y_scores]
        fp = find_fp(y_true, y_pred)
        print(fp)
        fp_percentage = (fp / len(y_pred)) * 100
        print(fp_percentage)
        threshold_list.append(threshold)
        fp_list.append(fp)
        if 0.9 < fp_percentage < 1:
            break

    while fp_percentage < 1:
        print("Fp percentage is under 1%")
        final_threshold -= final_threshold * 0.05
        print("New thesold: ", final_threshold)
        y_pred = [1.0 if score > final_threshold else 0.0 for score in y_scores]
        fp = find_fp(y_true, y_pred)
        print("FP after using the new thesold: ", fp)
        fp_percentage = (fp / len(y_pred)) * 100
        print("New fp percentage: ", fp_percentage)
        threshold_list.append(threshold)
        fp_list.append(fp)
        if 0.9 < fp_percentage < 1:
            break

    while fp_percentage > 1.1:
        print("Fp percentage is over 1%")
        final_threshold += final_threshold * 0.02
        print("New thesold: ", final_threshold)
        y_pred = [1.0 if score > final_threshold else 0.0 for score in y_scores]
        fp = find_fp(y_true, y_pred)
        print("FP after using the new thesold: ", fp)
        fp_percentage = (fp / len(y_pred)) * 100
        print("New fp percentage: ", fp_percentage)
        threshold_list.append(threshold)
        fp_list.append(fp)
        if 0.9 < fp_percentage < 1:
            break

    print("Final fp percentage:", fp_percentage)
    return final_threshold, threshold_list, fp_list


def find_max_score(y_scores):
    max_score = np.max(y_scores)
    return max_score




def process_graph(data):
    """
    Obtain the dense adjacency matrix of the graph.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Input graph.
    """
    data.s = to_dense_adj(data.edge_index)[0]


def compute_node_anomaly_score(x, x_, s, s_, alpha):
    """
    This function computes the anomaly score for each node in a given graph/batch.
    """

    score, _, _ = objective_function(x,
                                     x_,
                                     s,
                                     s_,
                                     alpha)

    return score.detach().cpu()


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

    return accuracy, precision, recall, f_score, tp, tn, fp, fn


def evaluate(args, model, dataset, compute_evaluation_metrics_flag, result_dict=None, compute_threshold=True):
    """
    This function evaluates the model on the IoT23 dataset and writes the results to a CSV file.

    Parameters
    ----------
    args: Command-line arguments or a configuration object containing parameters like dataset folder, graph type, etc.
    model: The model to be evaluated.

    Returns
    ----------
    None. Writes evaluation metrics to a CSV file.
    """
    # paths definition
    if 'val' in dataset:
        json_folder = args.json_folder.replace("split_test", "split")
    else:
        json_folder = args.json_folder
    json_set = os.path.join(json_folder, dataset)
    print(f"Dataset: {json_set}")
    threshold_folder = os.path.join(args.threshold_path, args.graph_type)

    opt_threshold = import_optimal_threshold(threshold_folder)
    print("Threshold to use: ", opt_threshold)

    # Dataloaders definition
    graph_set = Graph_dataset(dataset_path=args.dataset_folder,
                              json_path=json_set,
                              representation=args.graph_type,
                              normalize=args.normalize,
                              min_max=args.min_max)
    test_dataloader = DataLoader(graph_set, num_workers=0)

    # check if GPU is available
    if args.device == 'cuda':
        device = torch.device(args.device)

    total_time = 0.0
    score_list = []
    labels_list = []
    inference_time_list = []
    model.to(device)
    model.eval()
    print("Testing {} set...".format(dataset))
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            process_graph(batch)
            x = batch.x.to(device)
            s = batch.s.to(device)
            edge_index = batch.edge_index.to(device)
            start_time = time.time()
            x_, s_ = model(x, edge_index)
            end_time = time.time()
            anomaly_score, _, _ = objective_function(x, x_, s, s_, args.alpha)
            score_list.extend(anomaly_score.detach().cpu().numpy().tolist())
            labels_list.extend(batch.y.numpy().tolist())
            inference_time_list.extend([end_time - start_time])

            # process_graph(data)
            # x = data.x.to(device)
            # # print("X:", x)
            # s = data.s.to(device)
            # edge_index = data.edge_index.to(device)

            # start_time = time.time()  # seconds
            # x_, s_ = model(x, edge_index)
            # anomaly_score = compute_node_anomaly_score(
            #     x, x_, s, s_, args.alpha)
            # end_time = time.time()
            # score_list.extend(anomaly_score.numpy().tolist())
            # labels_list.extend(data.y.numpy().tolist())
            # elapsed_time = end_time - start_time
            # total_time += elapsed_time

    average_pred_time = total_time / len(test_dataloader)

    if 'val' in dataset and compute_threshold:
        max_score = find_max_score(score_list)
        threshold = max_score/2
        print("Starting threshold:", threshold)
        optimal_threshold, threshold_list, fp_list = find_optimal_threshold(
            labels_list, score_list, threshold)
        print("Optimal threshold to use in the next step: ", optimal_threshold)
        
        with open(os.path.join(threshold_folder, "new_optimal_threshold.txt"), "w") as file:        
            file.write(str(optimal_threshold))
        
    
    if compute_evaluation_metrics_flag:
        accuracy, precision, recall, fscore, tp, tn, fp, fn = compute_evaluation_metrics(
            opt_threshold, score_list, labels_list)
        test_writer.writerow([dataset, accuracy, precision,
                            recall, fscore, tp, tn, fp, fn, average_pred_time*1000])
    else:
        result_dict['score_list'] = score_list
        result_dict['labels_list'] = labels_list
        result_dict['inference_time'] = inference_time_list
        result_dict['opt_threshold'] = opt_threshold
        


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--hidden_dim',
                        type=int,
                        default=64,
                        help='Dimension of hidden embedding (default: 28)')

    parser.add_argument('--encoder_layers',
                        type=int,
                        default=3,
                        help='Number of encoder layers')

    parser.add_argument('--decoder_layers',
                        type=int,
                        default=2,
                        help='Number of decoder layers')

    parser.add_argument('--lr',
                        type=float,
                        default=5e-4,
                        help='Learning rate')

    parser.add_argument('--dropout',
                        type=float,
                        default=0.3,
                        help='Dropout rate')

    parser.add_argument('--alpha',
                        type=float,
                        default=0.7,
                        help='Balance parameter')

    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        help='GPU = cuda/CPU = cpu')

    parser.add_argument("--graph_type",
                        type=str,
                        default="etdg_graph",
                        help="Graph type to consider (similarity_graph/trajectory_graph/etdg_graph)")

    parser.add_argument("--checkpoint_path",
                        type=str,
                        default="/user/apaolillo/checkpoint_dir/dominant/DOMINANT_B4_64_NORM_150k_IoT23_etdg/checkpoints",
                        help="Folder from which take the model to evaluate")

    parser.add_argument("--dataset_folder",
                        type=str,
                        default="/user/apaolillo/Output_Grafi/150000/IoT_traces/base",
                        help="Dataset folder from which take the graphs")

    parser.add_argument("--json_folder",
                        type=str,
                        default="/user/apaolillo/Output_Grafi/split_test/150k/base/IoT_traces_dataset_split_etdg",
                        help="Dataset folder in json format from which take the dataset split")

    parser.add_argument("--result_path",
                        type=str,
                        default="/user/apaolillo/checkpoint_dir/dominant/DOMINANT_B4_64_NORM_150k_IoT23_etdg/y_pred_true",
                        help="Folder where to save test results")

    parser.add_argument("--threshold_path",
                        type=str,
                        default="/user/apaolillo/checkpoint_dir/dominant/DOMINANT_B4_64_NORM_150k_IoT23_etdg/thresholds",
                        help="Folder with optimal thresholds")

    parser.add_argument("--normalize",
                        type=int,
                        default=-1)
    
    parser.add_argument("--compute_evaluation_metrics_flag",
                        action='store_true')

    parser.add_argument("--dataset",
                        type=str,
                        default="IoT23")

    parser.add_argument("--min_max",
                        type=str,
                        default="/user/apaolillo/Output_Grafi/min_max_benign/150k/IoT23_min_max_benign")

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

    RESULT_DICT = OrderedDict()
    
    if not torch.cuda.is_available():
        print("CUDA is not available")
        exit()
        
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    csv_results_folder = os.path.join(args.result_path, args.graph_type)
    # create csv results folder if it does not exists
    if not os.path.exists(csv_results_folder):
        os.makedirs(csv_results_folder)
    
    if args.compute_evaluation_metrics_flag:
        test_result_file = open(os.path.join(
            csv_results_folder, f'result_{args.dataset}.csv'), "w", newline='')
        test_writer = csv.writer(test_result_file)
        test_writer.writerow(['Set', 'Accuracy', 'Precision',
                            'Recall', 'F-Score', "TP", "TN", "FP", "FN", 'Average pred time(ms)'])

    model = get_best_model(args, os.path.join(
        args.checkpoint_path, args.graph_type))

    #
    if args.dataset == "IoT23":
        datasets = ["val.json", "test_benign.json",
                    "test_malicious.json", "test_mixed.json"]
        # datasets = ["val.json"]
    elif args.dataset == "IoT_traces":
        datasets = ["test_benign.json"]
    elif args.dataset == "IoTID20":
        datasets = ["test_benign.json", "test_mixed.json"]

    seed_everything()

    for dataset in datasets:
        print(
            f"\n Starting the evaluation for {args.dataset}, {args.graph_type}, ...")
        if RESULT_DICT.get(args.dataset) is None:
            RESULT_DICT[args.dataset] = {}
        if RESULT_DICT[args.dataset].get(args.graph_type) is None:
            RESULT_DICT[args.dataset][args.graph_type] = {}
        
        RESULT_DICT[args.dataset][args.graph_type][dataset] = {}
            
        evaluate(args, 
                 model, 
                 dataset, 
                 args.compute_evaluation_metrics_flag, 
                 RESULT_DICT[args.dataset][args.graph_type][dataset])
    
    if args.compute_evaluation_metrics_flag:
        test_result_file.close()
    else:
        import json
        with open(os.path.join(csv_results_folder, f'predictions_result_{args.dataset}.json'), 'w') as file:
            json.dump(RESULT_DICT, file)
