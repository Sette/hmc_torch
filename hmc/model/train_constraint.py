
from hmc.model import ConstrainedFFNNModel, get_constr_out
from hmc.dataset import initialize_dataset

import os

import torch.distributed as dist

import argparse

import torch
import torch.utils.data
import torch.nn as nn

import random
import numpy as np
import networkx as nx
from tqdm import tqdm


from sklearn.impute import SimpleImputer

from sklearn import preprocessing


from sklearn.metrics import average_precision_score
# from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score, auc


def train_constraint():
    # Training settings
    parser = argparse.ArgumentParser(description='Train neural network')

    # Required  parameters
    parser.add_argument('--dataset', type=str, required=True, 
                        nargs='+', default=['seq_GO', 'derisi_GO', '0.6', '0.7'], 
                        help='List with dataset names to train')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='dataset path')
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
    parser.add_argument('--output_path', type=str, required=True,
                        help='output path')
    parser.add_argument('--device', type=int, default=0,
                        help='device (default:0)')
    parser.add_argument('--num_epochs', type=int, default=2000,
                        help='Max number of epochs to train (default:2000)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default:0)')
    
    args = parser.parse_args()
    hyperparams = {'batch_size': args.batch_size, 'num_layers': args.num_layers, 'dropout': args.dropout,
                   'non_lin': args.non_lin, 'hidden_dim': args.hidden_dim, 'lr': args.lr,
                   'weight_decay': args.weight_decay}
    ## Insert her a logic to use all datasets with arguments
    
    print(type(args.dataset))
    
    if 'all' in args.dataset:
        datasets = ['cellcycle_GO', 'derisi_GO', 'eisen_GO', 'expr_GO', 'gasch1_GO',
                    'gasch2_GO', 'seq_GO', 'spo_GO', 'cellcycle_FUN', 'derisi_FUN', 
                    'eisen_FUN', 'expr_FUN', 'gasch1_FUN', 'gasch2_FUN', 'seq_FUN', 'spo_FUN']
    else:
        if len(args.dataset) > 1:
            datasets = [str(dataset) for dataset in args.dataset]
        else:
            datasets = [args.dataset]
            assert ('_' in args.dataset)
            assert ('FUN' in args.dataset or 'GO' in args.dataset or 'others' in args.dataset)
        

    # Dictionaries with number of features and number of labels for each dataset
    input_dims = {'diatoms': 371, 'enron': 1001, 'imclef07a': 80, 'imclef07d': 80, 'cellcycle': 77, 'church': 27,
                  'derisi': 63, 'eisen': 79, 'expr': 560, 'gasch1': 173, 'gasch2': 52, 'hom': 47034, 'seq': 529,
                  'spo': 86}
    output_dims_FUN = {'cellcycle': 499, 'church': 499, 'derisi': 499, 'eisen': 461, 'expr': 499, 'gasch1': 499,
                       'gasch2': 499, 'hom': 499, 'seq': 499, 'spo': 499}
    output_dims_GO = {'cellcycle': 4122, 'church': 4122, 'derisi': 4116, 'eisen': 3570, 'expr': 4128, 'gasch1': 4122,
                      'gasch2': 4128, 'hom': 4128, 'seq': 4130, 'spo': 4116}
    output_dims_others = {'diatoms': 398, 'enron': 56, 'imclef07a': 96, 'imclef07d': 46, 'reuters': 102}
    output_dims = {'FUN': output_dims_FUN, 'GO': output_dims_GO, 'others': output_dims_others}

    # Set seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Verifica quantas GPUs estão disponíveis
    num_gpus = torch.cuda.device_count()
    print(f"Total de GPUs disponíveis: {num_gpus}")
    # Inicializa o processo
    if num_gpus > 1:
        # Configura variáveis de ambiente necessárias para DistributedDataParallel
        os.environ["MASTER_ADDR"] = "localhost"  # Ou o IP do nó mestre se for multi-nó
        os.environ["MASTER_PORT"] = "29500"  # Pode escolher outra porta se necessário
        os.environ["RANK"] = "0"  # Rank do processo principal
        os.environ["WORLD_SIZE"] = str(num_gpus)  # Total de processos

        # Inicializa o processo distribuído
        dist.init_process_group(backend='nccl', rank=0, world_size=num_gpus)
        
        # Obtém ID da GPU do processo atual
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    elif num_gpus == 1:
        device = torch.device(f'cuda:0')
    elif num_gpus == 0:
        device = torch.device('cpu')
        

    for dataset_name in datasets:
        print(".......................................")
        print("Experiment with {} dataset ".format(dataset_name))
        # Load train, val and test set
        data = dataset_name.split('_')[0]
        ontology = dataset_name.split('_')[1]
        train, val, test = initialize_dataset(dataset_name, args.dataset_path, output_path=args.output_path,  is_go=False)
        train.to_eval, val.to_eval, test.to_eval = torch.tensor(train.to_eval, dtype=torch.uint8), torch.tensor(
            val.to_eval, dtype=torch.uint8), torch.tensor(test.to_eval, dtype=torch.uint8)

        different_from_0 = torch.tensor(np.array((test.Y.sum(0) != 0), dtype=np.uint8), dtype=torch.uint8)

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
        # Transpose to get the ancestors for each node
        R = R.transpose(1, 0)
        R = R.unsqueeze(0).to(device)


        scaler = preprocessing.StandardScaler().fit(np.concatenate((train.X_cont, val.X_cont)))
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit(np.concatenate((train.X_cont,val.X_cont)))
        val.X_count, val.Y = scaler.transform(imp_mean.transform(val.X_cont)), torch.tensor(val.Y).to(
            device)
        train.X_count, train.Y = scaler.transform(imp_mean.transform(train.X_cont)), torch.tensor(
            train.Y).to(device)
        print(train.X_bin.shape)
        if train.X_bin.shape[0] > 0:
            train.X = np.concatenate([train.X_count, train.X_bin], axis=1)
            val.X = np.concatenate([val.X_count, val.X_bin], axis=1)
        else:
            train.X = train.X_count
            val.X = val.X_count

        train.X = torch.tensor(train.X).to(device)
        val.X = torch.tensor(val.X).to(device)

            # Create loaders
        train_dataset = [(x, y) for (x, y) in zip(train.X, train.Y)]

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
        model = ConstrainedFFNNModel(input_dims[data], args.hidden_dim, output_dims[ontology][data] + num_to_skip,
                                    hyperparams, R)
        model.to(device)
        # Usa DDP
        if num_gpus > 1:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.BCELoss()

        # Set patience
        patience, max_patience = 20, 20
        max_score = 0.0

        # Create folder for the dataset (if it does not exist)
        if not os.path.exists('logs/' + str(dataset_name) + '/'):
            os.makedirs('logs/' + str(dataset_name) + '/')
        for epoch in tqdm(range(num_epochs)):
            total_train = 0.0
            correct_train = 0.0
            model.train()

            train_score = 0

            for i, (x, labels) in enumerate(train_loader):
                x = x.to(device)
                labels = labels.to(device)

                # Clear gradients w.r.t. parameters
                optimizer.zero_grad()
                output = model(x.float())

                constr_output = get_constr_out(output, R)
                train_output = labels * output.double()
                train_output = get_constr_out(train_output, R)
                train_output = (1 - labels) * constr_output.double() + labels * train_output
                loss = criterion(train_output, labels)
                predicted = constr_output.data > 0.5
                # Total number of labels
                total_train += labels.size(0) * labels.size(1)
                # Total correct predictions
                correct_train += (predicted == labels.byte()).sum()

                # Getting gradients w.r.t. parameters
                loss.backward()
                # Updating parameters
                optimizer.step()

            model.eval()
            constr_output = constr_output.to('cpu')
            labels = labels.to('cpu')
            train_score = average_precision_score(labels, constr_output.data,
                                                average='micro')

            for i, (x, y) in enumerate(val_loader):
                x = x.to(device)
                y = y.to(device)

                constrained_output = model(x.float())
                predicted = constrained_output.data > 0.5
                # Total number of labels
                total = y.size(0) * y.size(1)
                # Total correct predictions
                correct = (predicted == y.byte()).sum()

                # Move output and label back to cpu to be processed by sklearn
                cpu_constrained_output = constrained_output.to('cpu')
                y = y.to('cpu')

                if i == 0:
                    constr_val = cpu_constrained_output
                    y_val = y
                else:
                    constr_val = torch.cat((constr_val, cpu_constrained_output), dim=0)
                    y_val = torch.cat((y_val, y), dim=0)

            score = average_precision_score(y_val, constr_val.data, average='micro')

            if score >= max_score:
                patience = max_patience
                max_score = score
            else:
                patience = patience - 1

            floss = open('logs/' + str(dataset_name) + '/measures_batch_size_' + str(args.batch_size) + '_lr_' + str(
                args.lr) + '_weight_decay_' + str(args.weight_decay) + '_seed_' + str(args.seed) + '_num_layers_' + str(
                args.num_layers) + '._hidden_dim_' + str(args.hidden_dim) + '_dropout_' + str(
                args.dropout) + '_' + args.non_lin, 'a')
            floss.write(
                '\nEpoch: {} - Loss: {:.4f}, Accuracy train: {:.5f}, Accuracy: {:.5f}, Precision score: ({:.5f})\n'.format(
                    epoch,
                    loss, float(correct_train) / float(total_train), float(correct) / float(total), score))
            floss.close()

            if patience == 0:
                break