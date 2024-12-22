import json
import os
import networkx as nx
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from hmc.model import HMCLocalClassificationModel, ConstrainedFFNNGlobalModel, get_constr_out
from hmc.dataset import HMCDataset
from hmc.model.losses import show_global_loss, show_local_losses
from hmc.utils.dir import create_job_id, create_dir
from hmc.model.arguments import get_parser
from hmc.model.metrics import generete_md

import numpy as np

import argparse

import torch
import torch.utils.data
import torch.nn as nn

import random

from sklearn.impute import SimpleImputer

from sklearn import preprocessing
from sklearn.model_selection import train_test_split


from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve, roc_auc_score, auc

def run():
    print("========================= PyTorch =========================")
    print("GPUs available: {}".format(torch.cuda.device_count()))
    print("===========================================================")

    job_id = create_job_id()
    print(f"Job ID: {job_id}")

    parser = get_parser()
    args = parser.parse_args()

    dropouts = [float(rate) for rate in args.dropouts]
    thresholds = [float(threshold) for threshold in args.thresholds]

    metadata_path = os.path.join(args.input_path, 'metadata.json')
    labels_path = os.path.join(args.input_path, 'labels.json')

    with open(metadata_path, 'r') as f:
        metadata = json.loads(f.read())

    with open(labels_path, 'r') as f:
        labels = json.loads(f.read())

    params = {
        'levels_size': labels['levels_size'],
        'input_size': metadata['sequence_size'],
        'dropouts': dropouts,
        'thresholds': thresholds
    }

    assert len(args.dropouts) == metadata['max_depth']
    assert len(args.lrs) == metadata['max_depth']

    model = HMCLocalClassificationModel(**params)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

    criterions = [nn.BCEWithLogitsLoss(reduction='sum') for _ in labels['levels_size']]

    if torch.cuda.is_available():
        model = model.to('cuda')
        criterions = [criterion.to('cuda') for criterion in criterions]
        

    torch_path = os.path.join(args.input_path, 'torch')
    metadata['train_torch_path'] = os.path.join(torch_path, 'train')
    metadata['val_torch_path'] = os.path.join(torch_path, 'val')
    metadata['test_torch_path'] = os.path.join(torch_path, 'test')

    ds_train = HMCDataset(metadata['train_torch_path'], params['levels_size'])
    ds_validation = HMCDataset(metadata['val_torch_path'], params['levels_size'])

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ds_validation, batch_size=args.batch_size, shuffle=False)

    assert isinstance(args.output_path, str)
    model_path: str = os.path.join(args.output_path, 'jobs' ,job_id)
    create_dir(model_path)

    early_stopping_patience = args.patience
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs+1):
        model.train()
        local_train_losses = [0.0 for _ in range(metadata['max_depth'])]
        for inputs, targets in train_loader:
            if torch.cuda.is_available():
                inputs, targets = inputs.to('cuda'), [target.to('cuda') for target in targets]
            outputs = model(inputs)
            
            # Zerar os gradientes antes de cada batch
            optimizer.zero_grad()

            total_loss = 0.0
            for index, (output, target) in enumerate(zip(outputs, targets)):
                loss = criterions[index](output, target)
                total_loss += loss
                local_train_losses[index] += loss.item()

        # Backward pass (cálculo dos gradientes)
        total_loss.backward()
        
        optimizer.step()

        local_train_losses = [loss / len(train_loader) for loss in local_train_losses]
        global_train_loss = sum(local_train_losses) / metadata['max_depth']

        print(f'Epoch {epoch}/{args.epochs}')
        show_local_losses(local_train_losses, set='Train')
        show_global_loss(global_train_loss, set='Train')
            
        model.eval()
        local_val_losses = [0.0 for _ in range(metadata['max_depth'])]
        with torch.no_grad():
            for inputs, targets in val_loader:
                if torch.cuda.is_available():
                    inputs, targets = inputs.to('cuda'), [target.to('cuda') for target in targets]
                outputs = model(inputs)

                total_val_loss = 0.0
                for index, (output, target) in enumerate(zip(outputs, targets)):
                    loss = criterions[index](output, target)
                    total_val_loss += loss
                    local_val_losses[index] += loss.item() 

        local_val_losses = [loss / len(val_loader) for loss in local_val_losses]
        global_val_loss = sum(local_val_losses) / metadata['max_depth']

        print(f'Epoch {epoch}/{args.epochs}')
        show_local_losses(local_val_losses, set='Val')
        show_global_loss(global_val_loss, set='Val')

        current_val_loss = round(global_val_loss, 4)
        if current_val_loss <= best_val_loss - 2e-4:
            best_val_loss = current_val_loss
            print('new best model')
            torch.save(model.state_dict(), os.path.join(model_path, f'best_binary-{epoch}.pth'))
        else:
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered")
                return None
            
            patience_counter += 1
    return None

def run_constrained():

    # Training settings
    parser = argparse.ArgumentParser(description='Train neural network')
    
    # Required  parameters
    parser.add_argument('--dataset', type=str, required=True,
                        help='dataset')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='input batch size for training')
    parser.add_argument('--lr', type=float, required=True,
                        help='learning rate')
    parser.add_argument('--dropout', type=float, required=True,
                        help='dropout probability')
    parser.add_argument('--hidden_dim', type=int, required=True,
                        help='size of the hidden layers')
    parser.add_argument('--num_layers', type=int, required=True,
                        help='number of hidden layers')
    parser.add_argument('--weight_decay', type=float, required=True,
                        help='weight decay')
    parser.add_argument('--non_lin', type=str, required=True,
                        help='non linearity function to be used in the hidden layers')

    # Other parameters 
    parser.add_argument('--device', type=int, default=0,
                        help='device (default:0)')
    parser.add_argument('--num_epochs', type=int, default=2000,
                        help='Max number of epochs to train (default:2000)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default:0)')

    args = parser.parse_args()
    hyperparams = {'batch_size':args.batch_size, 'num_layers':args.num_layers, 'dropout':args.dropout, 'non_lin':args.non_lin, 'hidden_dim':args.hidden_dim, 'lr':args.lr, 'weight_decay':args.weight_decay}

    assert('_' in args.dataset)
    assert('FUN' in args.dataset or 'GO' in args.dataset or 'others' in args.dataset)


    # Load train, val and test set
    dataset_name = args.dataset
    data = dataset_name.split('_')[0]
    ontology = dataset_name.split('_')[1]


   # Dictionaries with number of features and number of labels for each dataset
    input_dims = {'diatoms':371, 'enron':1001,'imclef07a': 80, 'imclef07d': 80,'cellcycle':77, 'church':27, 'derisi':63, 'eisen':79, 'expr':561, 'gasch1':173, 'gasch2':52, 'hom':47034, 'seq':529, 'spo':86}
    output_dims_FUN = {'cellcycle':499, 'church':499, 'derisi':499, 'eisen':461, 'expr':499, 'gasch1':499, 'gasch2':499, 'hom':499, 'seq':499, 'spo':499}
    output_dims_GO = {'cellcycle':4122, 'church':4122, 'derisi':4116, 'eisen':3570, 'expr':4128, 'gasch1':4122, 'gasch2':4128, 'hom':4128, 'seq':4130, 'spo':4116}
    output_dims_others = {'diatoms':398,'enron':56, 'imclef07a': 96, 'imclef07d': 46, 'reuters':102}
    output_dims = {'FUN':output_dims_FUN, 'GO':output_dims_GO, 'others':output_dims_others}


    # Set seed
    # Compute matrix of ancestors R
    # Given n classes, R is an (n x n) matrix where R_ij = 1 if class i is an ancestor of class j.

    # Create an n x n zero matrix R (same shape as train.A)
    R = np.zeros(train.A.shape)

    # Fill the diagonal with 1, because each class is considered its own ancestor
    np.fill_diagonal(R, 1)

    # Build a directed graph g from the adjacency matrix train.A
    # Here, train. A indicates parent-child relationships between classes.
    # nx.DiGraph(train.A) creates a directed graph where an edge i->j means class i is a parent (or ancestor) of class j.
    g = nx.DiGraph(train.A)

    # For each class i
    for i in range(len(train.A)):
        # Get the list of all descendants of class i in the graph g.
        # nx.descendants(g, i) returns all nodes reachable from i, thus all descendants of i.
        descendants = list(nx.descendants(g, i))
        if descendants:
            # Mark that i is an ancestor of all these descendants.
            # Setting R[i, descendants] = 1 means that in row i of R,
            # the columns corresponding to these descendants are set to 1.
            R[i, descendants] = 1

    # Convert the numpy array R to a PyTorch tensor
    R = torch.tensor(R)

    # Transpose R. Initially, R[i, j] = 1 means class i is an ancestor of class j.
    # By transposing, we have R[j, i] = 1 means class i is an ancestor of class j.
    # Depending on how the model output is indexed, this transpose might be required.
    R = R.transpose(1, 0)

    # Add an extra dimension at the start, making R of shape (1, n, n).
    # This can be useful for broadcasting in the code that uses R.
    R = R.unsqueeze(0).to(device)

    # Pick device
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")

    # Load the datasets
    if ('others' in args.dataset):
        #### to do load dataset
        #train, test = initialize_other_dataset(dataset_name, datasets)
        train, test = []
        train.to_eval, test.to_eval = torch.tensor(train.to_eval, dtype=torch.uint8),  torch.tensor(test.to_eval, dtype=torch.uint8)
        train.X, valX, train.Y, valY = train_test_split(train.X, train.Y, test_size=0.30, random_state=seed)
    else:
        #### to do load dataset
        train, val, test = []
        #train, val, test = initialize_dataset(dataset_name, datasets)
        train.to_eval, val.to_eval, test.to_eval = torch.tensor(train.to_eval, dtype=torch.uint8), torch.tensor(val.to_eval, dtype=torch.uint8), torch.tensor(test.to_eval, dtype=torch.uint8)
    

    # Compute matrix of ancestors R
    # Given n classes, R is an (n x n) matrix where R_ij = 1 if class i is ancestor of class j
    R = np.zeros(train.A.shape)
    np.fill_diagonal(R, 1)
    g = nx.DiGraph(train.A)
    for i in range(len(train.A)):
        descendants = list(nx.descendants(g, i))
        if descendants:
            R[i, descendants] = 1
    R = torch.tensor(R)
    #Transpose to get the ancestors for each node 
    R = R.transpose(1, 0)
    R = R.unsqueeze(0).to(device)


    # Rescale dataset and impute missing data
    if ('others' in args.dataset):
        scaler = preprocessing.StandardScaler().fit((train.X))
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit((train.X))
        train.X, train.Y = torch.tensor(scaler.transform(imp_mean.transform(train.X))).to(device), torch.tensor(train.Y).to(device)
        valX, valY = torch.tensor(scaler.transform(imp_mean.transform(valX))).to(device), torch.tensor(valY).to(device)
    else:
        scaler = preprocessing.StandardScaler().fit(np.concatenate((train.X, val.X)))
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit(np.concatenate((train.X, val.X)))
        val.X, val.Y = torch.tensor(scaler.transform(imp_mean.transform(val.X))).to(device), torch.tensor(val.Y).to(device)
        train.X, train.Y = torch.tensor(scaler.transform(imp_mean.transform(train.X))).to(device), torch.tensor(train.Y).to(device)        

    # Create loaders 
    train_dataset = [(x, y) for (x, y) in zip(train.X, train.Y)]
    if ('others' in args.dataset):
        val_dataset = [(x, y) for (x, y) in zip(valX, valY)]
    else:
        val_dataset = [(x, y) for (x, y) in zip(val.X, val.Y)]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=args.batch_size, 
                                            shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                            batch_size=args.batch_size, 
                                            shuffle=False)


    num_epochs = args.num_epochs
    if 'GO' in dataset_name: 
        num_to_skip = 4
    else:
        num_to_skip = 1 

    # Create the model
    model = ConstrainedFFNNGlobalModel(input_dims[data], args.hidden_dim, output_dims[ontology][data]+num_to_skip, hyperparams, R)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
    criterion = nn.BCELoss()

    # Set patience 
    patience, max_patience = 20, 20
    max_score = 0.0

    # Create folder for the dataset (if it does not exist)
    if not os.path.exists('logs/'+str(dataset_name)+'/'):
         os.makedirs('logs/'+str(dataset_name)+'/')

    for epoch in range(num_epochs):
        total_train = 0.0
        correct_train = 0.0
        model.train()

        train_score = 0

        for i, (x, labels) in enumerate(train_loader):
            x = x.to(device)
            labels = labels.to(device)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass: compute the model output
            output = model(x.float())

            # Apply hierarchical constraints to the raw output
            constr_output = get_constr_out(output, R)

            # Here, we combine labels and output in a particular way:
            # labels * output: for positions where label=1, keep output; where label=0, set output=0.
            train_output = labels * output.double()

            # Apply constraints again to the filtered output
            train_output = get_constr_out(train_output, R)

            # Recombine the constrained outputs:
            # (1 - labels)*constr_output + labels*train_output means:
            # - Where label=0, we use constr_output
            # - Where label=1, we use train_output (the already filtered and constrained one)
            train_output = (1 - labels) * constr_output.double() + labels * train_output

            # Compute the loss only on selected output indices (train.to_eval)
            loss = criterion(train_output[:, train.to_eval], labels[:, train.to_eval])

            # Make binary predictions (threshold at 0.5)
            predicted = constr_output.data > 0.5

            # Update training statistics:
            # total number of labels (instances * number of classes)
            total_train += labels.size(0) * labels.size(1)
            # count correct predictions
            correct_train += (predicted == labels.byte()).sum()

            # Backpropagation
            loss.backward()
            # Parameter update
            optimizer.step()

        # Evaluation mode
        model.eval()

        # Move tensors to CPU for further processing, evaluation, or logging
        constr_output = constr_output.to('cpu')
        train_score = average_precision_score(labels[:,train.to_eval], constr_output.data[:,train.to_eval], average='micro') 

        for i, (x,y) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)

            constrained_output = model(x.float())
            predicted = constrained_output.data > 0.5
            # Total number of labels
            total = y.size(0) * y.size(1)
            # Total correct predictions
            correct = (predicted == y.byte()).sum()

            #Move output and label back to cpu to be processed by sklearn
            cpu_constrained_output = constrained_output.to('cpu')
            y = y.to('cpu')

            if i == 0:
                constr_val = cpu_constrained_output
                y_val = y
            else:
                constr_val = torch.cat((constr_val, cpu_constrained_output), dim=0)
                y_val = torch.cat((y_val, y), dim =0)

        score = average_precision_score(y_val[:,train.to_eval], constr_val.data[:,train.to_eval], average='micro') 
        
        if score >= max_score:
            patience = max_patience
            max_score = score
        else:
            patience = patience - 1
        
        floss= open('logs/'+str(dataset_name)+'/measures_batch_size_'+str(args.batch_size)+'_lr_'+str(args.lr)+'_weight_decay_'+str(args.weight_decay)+'_seed_'+str(args.seed)+'_num_layers_'+str(args.num_layers)+'._hidden_dim_'+str(args.hidden_dim)+'_dropout_'+str(args.dropout)+'_'+args.non_lin, 'a')
        floss.write('\nEpoch: {} - Loss: {:.4f}, Accuracy train: {:.5f}, Accuracy: {:.5f}, Precision score: ({:.5f})\n'.format(epoch,
                    loss, float(correct_train)/float(total_train), float(correct)/float(total), score))
        floss.close()

        if patience == 0:
            break


