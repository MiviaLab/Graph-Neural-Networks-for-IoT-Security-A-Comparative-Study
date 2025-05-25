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

    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c



def update_radius(dist, beta):
    with torch.no_grad():
        radius = torch.quantile(torch.sqrt(dist), 1 - beta)
    return radius

def train(args):
    # paths definition
    trainingset_folder = os.path.join(args.dataset_folder, args.fold, "train")
    validationset_folder = os.path.join(args.dataset_folder, args.fold, "validation")
    csv_results_folder = os.path.join(args.csv_results_folder, args.fold, args.graph_type)
    checkpoints_folder = os.path.join(args.checkpoint_folder, args.fold, args.graph_type)

    center_folder = os.path.join(args.center_path, args.fold, args.graph_type)
    radius_folder = os.path.join(args.radius_path, args.fold, args.graph_type)

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
    train_result_file = open(os.path.join(csv_results_folder, 'train_result.csv'), "w", newline='')
    train_writer = csv.writer(train_result_file)
    val_result_file = open(os.path.join(csv_results_folder, 'validation_result.csv'), "w", newline='')
    val_writer = csv.writer(val_result_file)
    train_writer.writerow(['epoch','loss', 'time_per_epoch'])
    val_writer.writerow(['epoch','loss'])
    
    # Dataloaders definition
    train = Graph_dataset(trainingset_folder, args.graph_type)
    val = Graph_dataset(validationset_folder, args.graph_type)
    train_dataloader = DataLoader(train, batch_size=args.batch_size, num_workers=0, shuffle = True)
    val_dataloader = DataLoader(val, batch_size=args.batch_size, num_workers=0)

    # check if GPU is available
    if args.device == 'cuda':
        device = torch.device(args.device)

    if args.graph_type == "similarity_graph" or "trajectory_graph":
        in_dim = 57
    if args.graph_type == "etdg_graph":
        in_dim = 72
    
    # model definition and loading to GPU
    model = OCGNN(in_dim = in_dim,
                 hid_dim = args.hid_dim,
                 num_layers = args.num_layers,
                 dropout = args.dropout,
                 beta = args.beta,
                 backbone = GraphSAGE).to(device)
    
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    center = init_center(args, train_dataloader, model, args.eps)
    torch.save(center, os.path.join(center_folder, 'center.pt'))


    # start training
    best_val_loss = float("inf")
    best_epoch = 0
    consecutive_bad_epochs = 0
    radius = 0.0
    dv_tr = []
    for epoch in range(args.epochs):
        start_time = time.time()
        print("\n ------- Epoch ", epoch, " - at: ", start_time)
        model.train(True)

        for batch in train_dataloader:
            # Forward pass
            optimizer.zero_grad()
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)

            emb = model(x, edge_index)

            loss, _, dist = model.loss_func(emb, center, radius)
            dv_tr.append(dist)
    
            # the radius will be updated each 5 epochs
            if epoch % 5 == 0:
                print("Epoch:", epoch)
                final_dist = torch.cat(dv_tr, dim=0)
                print("Final dist:", final_dist)
                radius = update_radius(final_dist, args.beta)
                torch.save(radius, os.path.join(radius_folder, 'radius.pt'))
                radius = radius.to(args.device)
                print("Radius:", radius)
                dv_tr = []
                break

            # Backpropagation
            loss.backward()
            optimizer.step()
        
        model.train(False)

        print("\n Computing training results...")
        total_train_loss = 0.0
        with torch.no_grad():
            for batch in train_dataloader:
                x = batch.x.to(device)
                edge_index = batch.edge_index.to(device)

                emb = model(x, edge_index)

                train_loss, _, _ = model.loss_func(emb, center, radius)
                end_time = time.time()
                
                total_train_loss += torch.mean(train_loss.detach().cpu()).item()

            average_train_loss = total_train_loss / len(train_dataloader)
            
            elapsed_time = end_time - start_time

            train_writer.writerow([epoch, average_train_loss, elapsed_time])

            print("\nTrain Loss:", average_train_loss,
                " - elapsed time: ", elapsed_time)


            # Validation
            print("\n Validating...")
            total_val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for batch in val_dataloader:
                    x = batch.x.to(device)
                    edge_index = batch.edge_index.to(device)

                    emb = model(x, edge_index)

                    val_loss, _, _ = model.loss_func(emb, center, radius)
                    end_time = time.time()

                    total_val_loss += torch.mean(val_loss.detach().cpu()).item()

                average_val_loss = total_val_loss / len(val_dataloader)
                
                val_writer.writerow([epoch, average_val_loss])

                print("\nVal Loss:", average_val_loss)
                

                if average_val_loss < best_val_loss:
                    best_epoch = epoch
                    best_val_loss = average_val_loss
                    torch.save(model.state_dict(), checkpoints_folder + "/OCGNN_model_{}".format(best_epoch))
                    consecutive_bad_epochs = 0
                else:
                    consecutive_bad_epochs += 1
                
                if consecutive_bad_epochs == args.patience:
                    print(f'Validation loss has not improved for {args.patience} epochs. Early stopping...')
                    break 

    val_result_file.close() 
    train_result_file.close()




if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',
                        type = int,
                        default = 80,
                        help='Training epochs')

    parser.add_argument('--patience',
                        type = int,
                        default = 10,
                        help='Number of epoch for early stopping')
    
    parser.add_argument('--batch_size',
                        type = int,
                        default = 4,
                        help='Number of graphs per epoch')   
     
    parser.add_argument('--hid_dim',
                        type = int,
                        default=28,
                        help='Dimension of hidden embedding (default: 28)')
    
    parser.add_argument('--num_layers',
                        type = int,
                        default = 2,
                        help = 'Number of network layers')
    
    parser.add_argument('--dropout',
                        type = float,
                        default = 0.5,
                        help='Dropout rate')
    
    parser.add_argument('--lr',
                        type = float,
                        default = 3e-4,
                        help='Learning rate')

    parser.add_argument('--beta',
                        type = float,
                        default = 0.1,
                        help='Hyperparam to control the trade-off between the sphere volume and the penalties')

    parser.add_argument('--eps',
                        type = float,
                        default = 0.01,
                        help='Slack parameter (float)')


    parser.add_argument('--weight_decay',
                        type = float,
                        default = 5e-4,
                        help='Weight for L2 loss')

    parser.add_argument('--device',
                        type = str,
                        default = 'cuda',
                        help='GPU = cuda/CPU = cpu')

    parser.add_argument("--dataset_folder",
                        type = str,
                        default = "/user/sranieri/Thesis/Dataset/IoT23/folds",
                        help="Dataset folder from which take the sets")
    
    parser.add_argument("--graph_type",
                        type = str,
                        default = "trajectory_graph",
                        help = "Graph type to consider")

    parser.add_argument("--fold",
                        type = str,
                        default = "fold0",
                        help="Dataset fold as validation set")

    
    parser.add_argument("--checkpoint_folder",
                    type = str,
                    default = "/user/sranieri/Thesis/Results/OCGNN/checkpoints/folds",
                    help="Folder where to save checkpoints")

    parser.add_argument("--csv_results_folder",
                    type = str,
                    default = "/user/sranieri/Thesis/Results/OCGNN/csv_results/folds",
                    help="Folder where to save results as csv file")

    parser.add_argument("--center_path",
                        type=str,
                        default="/user/sranieri/Thesis/Results/OCGNN/center/folds",
                        help="Folder where to save center value")

    parser.add_argument("--radius_path",
                        type=str,
                        default="/user/sranieri/Thesis/Results/OCGNN/radius/folds",
                        help="Folder where to save radius value")
    
    args = parser.parse_args()
    

    train(args)