import torch
import os
import csv
import argparse
import time
import pickle as pk
from DOMINANT import DOMINANT
from functional import objective_function
from torch_geometric.utils import to_dense_adj
from torch_geometric.loader import DataLoader
from Graph_dataset import Graph_dataset



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

    score, train_attr_error, train_struct_error = objective_function(x,
                                                                    x_,
                                                                    s,
                                                                    s_,
                                                                    alpha)

    return score, train_attr_error.detach().cpu(), train_struct_error.detach().cpu()



def train(args):
    # paths definition
    trainingset_folder = os.path.join(args.dataset_folder, "train")
    validationset_folder = os.path.join(args.dataset_folder, "folds", args.fold, "validation")
    csv_results_folder = os.path.join(args.csv_results_folder, args.fold, args.graph_type)
    checkpoints_folder = os.path.join(args.checkpoint_folder, args.fold, args.graph_type)

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
    train_writer.writerow(['epoch','loss','struct_loss','feat_loss', 'time_per_epoch'])
    val_writer.writerow(['epoch','loss','struct_loss','feat_loss'])
    
    # Dataloaders definition
    train = Graph_dataset(trainingset_folder, args.graph_type)
    val = Graph_dataset(validationset_folder, args.graph_type)
    train_dataloader = DataLoader(train, batch_size = args.batch_size, num_workers=0, shuffle = True)
    val_dataloader = DataLoader(val, batch_size = args.batch_size, num_workers=0)

    # check if GPU is available
    if args.device == 'cuda':
        device = torch.device(args.device)

    if args.graph_type == "similarity_graph" or "trajectory_graph":
        in_dim = 57
    if args.graph_type == "etdg_graph":
        in_dim = 72
    # model definition and loading to GPU
    model = DOMINANT(in_dim = in_dim,
                    hid_dim = args.hidden_dim,
                    encoder_layers = args.encoder_layers,
                    decoder_layers = args.decoder_layers,
                    dropout = args.dropout).to(device)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    # start training
    best_val_loss = float("inf")
    best_epoch = 0
    consecutive_bad_epochs = 0
    for epoch in range(args.epochs):
        start_time = time.time()
        print("\n ------- Epoch ", epoch, " - at: ", start_time)
        model.train(True)
        for batch in train_dataloader:
            # Obtain the dense adjacency matrix of the graph
            process_graph(batch)
            # Forward pass
            x = batch.x.to(device)
            s = batch.s.to(device)
            edge_index = batch.edge_index.to(device)

            x_, s_ = model(x, edge_index)

            # Calculate loss
            score, _, _ = compute_node_anomaly_score(x, x_, s, s_, args.alpha)

            # Backpropagation
            optimizer.zero_grad()
            loss = torch.mean(score)
            loss.backward()
            optimizer.step()
        
        model.train(False)

        print("\n Computing training results...")
        total_train_loss = 0.0
        total_train_attr_error = 0.0
        total_train_struct_error = 0.0
        with torch.no_grad():
            for batch in train_dataloader:
                process_graph(batch)
                x = batch.x.to(device)
                s = batch.s.to(device)
                edge_index = batch.edge_index.to(device)

                x_, s_ = model(x, edge_index)

                train_loss, train_attr_error, train_struct_error = compute_node_anomaly_score(x, x_, s, s_, args.alpha)
                end_time = time.time()
                total_train_loss += torch.mean(train_loss.detach().cpu()).item()
                total_train_attr_error += torch.mean(train_attr_error.detach().cpu()).item()
                total_train_struct_error += torch.mean(train_struct_error.detach().cpu()).item()


            average_train_loss = total_train_loss / len(train_dataloader)
            average_attr_error = total_train_attr_error / len(train_dataloader)
            average_struct_error = total_train_struct_error / len(train_dataloader)
            
            elapsed_time = end_time - start_time

            train_writer.writerow([epoch, average_train_loss, average_attr_error, average_struct_error, elapsed_time])

            print("\nTrain Loss:", average_train_loss,
                " - Train Struct Loss:", average_attr_error,
                " - Train Feat Loss:", average_struct_error,
                " - elapsed time: ", elapsed_time)


            # Validation
            print("\n Validating...")
            total_val_loss = 0.0
            total_val_attr_error = 0.0
            total_val_struct_error = 0.0
            model.eval()
            with torch.no_grad():
                for batch in val_dataloader:
                    process_graph(batch)
                    x = batch.x.to(device)
                    s = batch.s.to(device)
                    edge_index = batch.edge_index.to(device)

                    x_, s_ = model(x, edge_index)

                    # Calculate loss
                    val_loss, val_attr_error, val_struct_error = compute_node_anomaly_score(x, x_, s, s_, args.alpha)

                    total_val_loss += torch.mean(val_loss.detach().cpu()).item()
                    total_val_attr_error += torch.mean(val_attr_error.detach().cpu()).item()
                    total_val_struct_error += torch.mean(val_struct_error.detach().cpu()).item()

                average_val_loss = total_val_loss / len(val_dataloader)
                average_val_error = total_val_attr_error / len(val_dataloader)
                average_val_error = total_val_struct_error / len(val_dataloader)
                
                val_writer.writerow([epoch, average_val_loss, average_val_error, average_val_error])

                print("\nVal Loss:", average_val_loss,
                    " - Val Struct Loss:", average_val_error,
                    " - Val Feat Loss:", average_val_error)
                

                if average_val_loss < best_val_loss:
                    best_epoch = epoch
                    best_val_loss = average_val_loss
                    torch.save(model.state_dict(), checkpoints_folder + "/DOMINANT_model_{}".format(best_epoch))
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
                        default = 100,
                        help='Training epochs')

    parser.add_argument('--patience',
                        type = int,
                        default = 20,
                        help='Number of epoch for early stopping')
    
    parser.add_argument('--batch_size',
                        type = int,
                        default = 4,
                        help='Number of graphs per epoch (not used for trajectory graph representation)')   
     
    parser.add_argument('--hidden_dim',
                        type = int,
                        default=28,
                        help='Dimension of hidden embedding (default: 28)')
    
    parser.add_argument('--encoder_layers',
                        type = int,
                        default = 3,
                        help = 'Number of encoder layers')

    parser.add_argument('--decoder_layers',
                        type = int,
                        default = 2,
                        help = 'Number of decoder layers')
    
    parser.add_argument('--lr',
                        type = float,
                        default = 5e-4,
                        help='Learning rate')

    parser.add_argument('--dropout',
                        type = float,
                        default = 0.3,
                        help='Dropout rate')

    parser.add_argument('--alpha',
                        type = float,
                        default = 0.7,
                        help='Balance parameter')

    parser.add_argument('--device',
                        type = str,
                        default = 'cuda',
                        help='GPU = cuda/CPU = cpu')

    parser.add_argument("--dataset_folder",
                        type = str,
                        default = "/user/sranieri/Thesis/Dataset/IoT23",
                        help="Dataset folder from which take the sets")
    
    parser.add_argument("--fold",
                        type = str,
                        default = "fold1",
                        help="Dataset fold as validation set")
    
    parser.add_argument("--graph_type",
                        type = str,
                        default = "similarity_graph",
                        help = "Graph type to consider")
    
    parser.add_argument("--checkpoint_folder",
                    type = str,
                    default = "/user/sranieri/Thesis/Results/DOMINANT/checkpoints/folds",
                    help="Folder where to save checkpoints")

    parser.add_argument("--csv_results_folder",
                    type = str,
                    default = "/user/sranieri/Thesis/Results/DOMINANT/csv_results/folds",
                    help="Folder where to save results as csv file")
    args = parser.parse_args()
    
    """
    import debugpy
    debugpy.listen(('0.0.0.0', 5678))
    print("Waiting for debugger")
    debugpy.wait_for_client() """
    train(args)