import torch
import os
import csv
import argparse
import numpy as np
import time
from OCGNN import OCGNN
from torch_geometric.nn import GraphSAGE
from torch_geometric.loader import DataLoader
from Graph_dataset import Graph_dataset
import wandb
import warnings
import random
from test import compute_evaluation_metrics
warnings.filterwarnings("ignore")


def seed_everything(seed=42):
    # Set the seed for the built-in random module
    random.seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)

    # Set the seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_center(args, train_dataloader, model, eps=0.001):
    """
    Calculate the hypersphere center 'c' as the mean embedding from an initial forward pass over the data.

    Parameters:
    - args: A namespace or an object that contains the necessary arguments.
        - args.hid_dim: Integer. Dimension of the embeddings (hidden dimension).
        - args.device: String. Device type ('cuda' or 'cpu') on which computations will be performed.
    - train_dataloader: DataLoader. Provides batches of training data.
    - model: PyTorch model. Used to compute embeddings for the input data.
    - eps: Float, optional (default is 0.001). A small value to adjust embeddings too close to zero.

    Returns:
    - c: torch.Tensor. The center of the hypersphere initialized as the mean of the embeddings.
        It is a 1D tensor of shape (args.hid_dim,).

    Notes:
    The function computes embeddings for each batch of data, concatenates them, and then calculates the mean
    across all embeddings to get the hypersphere center 'c'. If any value in 'c' is too close to zero, it's 
    adjusted by `eps` to ensure that a zero unit doesn't trivially match with zero weights.
    """

    n_samples = 0
    c = torch.zeros(args.hid_dim).to(args.device)

    model.eval()
    embeddings = []

    with torch.no_grad():
        for data in train_dataloader:
            x = data.x.to(args.device)
            edge_index = data.edge_index.to(args.device)
            emb = model(x, edge_index)
            embeddings.append(emb)

    final_tensor = torch.cat(embeddings, dim=0)
    n_samples = final_tensor.shape[0]
    c = torch.sum(final_tensor, dim=0)

    c /= n_samples
    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c


def update_radius(dist, beta):
    with torch.no_grad():
        radius = torch.quantile(torch.sqrt(dist), 1 - beta)
    return radius


def train(args):
    seed_everything()
    # paths definition
    csv_results_folder = os.path.join(args.csv_results_folder, args.graph_type)
    checkpoints_folder = os.path.join(args.checkpoint_folder, args.graph_type)
    center_folder = os.path.join(args.center_path, args.graph_type)
    radius_folder = os.path.join(args.radius_path, args.graph_type)

    if not os.path.exists(radius_folder):
        os.makedirs(radius_folder)

    if not os.path.exists(center_folder):
        os.makedirs(center_folder)
    # create csv results folder if it does not exists
    if not os.path.exists(csv_results_folder):
        os.makedirs(csv_results_folder)

    # create checkpoint folder if it does not exists
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)

    # write training and validation results in a csv file
    train_result_file = open(os.path.join(
        csv_results_folder, 'train_result.csv'), "w", newline='')
    train_writer = csv.writer(train_result_file)
    val_result_file = open(os.path.join(
        csv_results_folder, 'validation_result.csv'), "w", newline='')
    val_writer = csv.writer(val_result_file)
    train_writer.writerow(['epoch', 'loss', 'time_per_epoch'])
    val_writer.writerow(['epoch', 'loss'])

    # Dataloaders definition
    json_training_set = os.path.join(args.json_folder, "train.json")
    json_validation_set = os.path.join(args.json_folder, "val.json")

    # dataset_path, json_path, representation
    train = Graph_dataset(args.dataset_folder,
                          json_training_set,
                          args.graph_type,
                          args.normalize,
                          args.min_max)
    val = Graph_dataset(args.dataset_folder,
                        json_validation_set,
                        args.graph_type,
                        args.normalize,
                        args.min_max)
    train_dataloader = DataLoader(
        train, batch_size=args.batch_size, num_workers=8, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=args.batch_size, num_workers=8)

    # check if GPU is available
    if args.device == 'cuda':
        device = torch.device(args.device)

    if args.graph_type == "tdg_graph":
        in_dim = 57
    if args.graph_type == "etdg_graph":
        in_dim = 72  # 15  # 72
    if args.graph_type == "only_etdg_graph":
        in_dim = 15
    if args.graph_type == "precision_features":
        pass

    # model definition and loading to GPU
    model = OCGNN(in_dim=in_dim,
                  hid_dim=args.hid_dim,
                  num_layers=args.num_layers,
                  dropout=args.dropout,
                  beta=args.beta,
                  backbone=GraphSAGE).to(device)

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    center = init_center(args, train_dataloader, model, args.eps)
    # print("Center: ", center)
    torch.save(center, os.path.join(center_folder, 'center.pt'))

    # start training
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_epoch = 0
    consecutive_bad_epochs = 0
    radius = 0.0
    dv_tr = []
    to_log = dict()
    step = 0
    for epoch in range(args.epochs):
        start_time = time.time()
        print("\n ------- Epoch ", epoch, " - at: ", start_time)
        model.train(True)

        dv_tr = []
        for batch in train_dataloader:

            step += 1
            # Forward pass
            optimizer.zero_grad()
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)

            emb = model(x, edge_index)

            loss, _, dist = model.loss_func(emb, center, radius)
            dv_tr.append(dist)

            if args.wandb_log:
                to_log["train_step/step"] = step
                to_log["train_step/loss"] = loss
                wandb.log(to_log)

            # if epoch % 5 == 0:
            #     # at the end of epoch update radius
            #     final_dist = torch.cat(dv_tr, dim=0)
            #     radius = update_radius(final_dist, args.beta)
            #     torch.save(radius, os.path.join(
            #         radius_folder, f'radius.pt'))
            #     radius = radius.to(args.device)
            #     print("Epoch:", epoch)
            #     print("Final dist:", final_dist)
            #     print("Radius:", radius)

            # Backpropagation
            loss.backward()
            optimizer.step()

        if args.wandb_log:
            to_log["train_step/radius"] = radius
            to_log["train_step/epoch"] = epoch
            wandb.log(to_log)

        model.train(False)

        # at the end of epoch update radius
        final_dist = torch.cat(dv_tr, dim=0)
        radius = update_radius(final_dist, args.beta)
        torch.save(radius, os.path.join(
            radius_folder, f'radius_{epoch}.pt'))
        radius = radius.to(args.device)
        print("Epoch:", epoch)
        print("Final dist:", final_dist)
        print("Radius:", radius)

        print("\n Computing training results...")
        total_train_loss = 0.0
        with torch.no_grad():
            for batch in train_dataloader:
                x = batch.x.to(device)
                edge_index = batch.edge_index.to(device)

                emb = model(x, edge_index)

                train_loss, _, _ = model.loss_func(emb, center, radius)
                end_time = time.time()

                total_train_loss += torch.mean(
                    train_loss.detach().cpu()).item()

            average_train_loss = total_train_loss / len(train_dataloader)

            elapsed_time = end_time - start_time

            train_writer.writerow([epoch, average_train_loss, elapsed_time])

            print("\nTrain Loss:", average_train_loss)
            if args.wandb_log:
                to_log["train_epoch/loss"] = average_train_loss
                to_log["train_epoch/epoch"] = epoch
                wandb.log(to_log)
            # Validation
            print("\n Validating...")
            total_val_loss = 0.0
            total_val_tp = 0
            total_val_tn = 0
            total_val_fp = 0
            total_val_fn = 0
            total_number_samples = 0
            best_threshold = .0
            model.eval()
            with torch.no_grad():
                for batch in val_dataloader:
                    x = batch.x.to(device)
                    edge_index = batch.edge_index.to(device)

                    emb = model(x, edge_index)

                    val_loss, score, _ = model.loss_func(emb, center, radius)
                
                    _, _, _, _, tp, tn, fp, fn = compute_evaluation_metrics(
                        threshold=best_threshold, 
                        y_scores = score, 
                        y_true = batch.y)
                    total_number_samples += len(batch.y)
                    
                    total_val_tp += tp
                    total_val_tn += tn
                    total_val_fp += fp
                    total_val_fn += fn
                
                    total_val_loss += torch.mean(
                        val_loss.detach().cpu()).item()

                average_val_loss = total_val_loss / len(val_dataloader)

                val_writer.writerow([epoch, average_val_loss])

                accuracy = (total_val_tp + total_val_tn) / total_number_samples
                
                print("\nVal Loss:", average_val_loss)
                if args.wandb_log:
                    to_log["val_epoch/loss"] = average_val_loss
                    to_log["val_epoch/epoch"] = epoch
                    wandb.log(to_log)
                
                if accuracy > best_val_accuracy and accuracy < 0.99:
                    best_epoch = epoch
                    best_val_accuracy = accuracy
                    print(f"Saving checkpoit OCGNN_model_{best_epoch}")
                    torch.save(model.state_dict(), checkpoints_folder +
                               "/OCGNN_model_{}".format(best_epoch))
                    consecutive_bad_epochs = 0
                else:
                    consecutive_bad_epochs += 1
                
                # if average_val_loss < best_val_loss:
                #     best_epoch = epoch
                #     best_val_loss = average_val_loss
                #     print(f"Saving checkpoit OCGNN_model_{best_epoch}")
                #     print(f"{checkpoints_folder}")
                #     torch.save(model.state_dict(), checkpoints_folder +
                #                "/OCGNN_model_{}".format(best_epoch))
                #     consecutive_bad_epochs = 0
                # else:
                #     consecutive_bad_epochs += 1

                if consecutive_bad_epochs == args.patience:
                    print(
                        f'Validation loss has not improved for {args.patience} epochs. Early stopping...')
                    break

    val_result_file.close()
    train_result_file.close()


if __name__ == "__main__":
    # 60 sim e etd
    # traj 80
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',
                        type=int,
                        default=150,
                        help='Training epochs')

    parser.add_argument('--patience',
                        type=int,
                        default=15,
                        help='Number of epoch for early stopping')

    parser.add_argument('--batch_size',
                        type=int,
                        default=4,
                        help='Number of graphs per epoch')

    parser.add_argument('--hid_dim',
                        type=int,
                        default=64,
                        help='Dimension of hidden embedding (default: 28)')

    parser.add_argument('--num_layers',
                        type=int,
                        default=2,
                        help='Number of network layers')

    parser.add_argument('--dropout',
                        type=float,
                        default=0.5,
                        help='Dropout rate')

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

    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        help='GPU = cuda/CPU = cpu')

    parser.add_argument("--dataset_folder",
                        type=str,
                        default="/user/sranieri/IoT23_graphs",
                        help="Dataset folder from which take the graphs")

    parser.add_argument("--json_folder",
                        type=str,
                        default="/user/sranieri/BORSA/Dataset",
                        help="Dataset folder in json format from which take the dataset split")

    parser.add_argument("--graph_type",
                        type=str,
                        default="etdg_graph",
                        help="Graph type to consider (similarity_graph/trajectory_graph/etdg_graph)")

    parser.add_argument("--checkpoint_folder",
                        type=str,
                        default="/user/sranieri/BORSA/Results/OCGNN_new/checkpoints",
                        help="Folder where to save checkpoints")

    parser.add_argument("--csv_results_folder",
                        type=str,
                        default="/user/sranieri/BORSA/Results/OCGNN_new/csv_results",
                        help="Folder where to save results as csv file")

    parser.add_argument("--center_path",
                        type=str,
                        default="/user/sranieri/BORSA/Results/OCGNN_new/center",
                        help="Folder where to save center value")

    parser.add_argument("--radius_path",
                        type=str,
                        default="/user/sranieri/BORSA/Results/OCGNN_new/radius",
                        help="Folder where to save radius value")

    parser.add_argument("--model",
                        type=str)

    parser.add_argument("--normalize",
                        type=int,
                        default=1)

    parser.add_argument("--min_max",
                        type=str,
                        default=None)

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

    if args.wandb_log:
        # init wandb_log
        project_name = f"{args.model}"
        run = wandb.init(project=project_name,
                         name=project_name,
                         sync_tensorboard=False)

    train(args)
