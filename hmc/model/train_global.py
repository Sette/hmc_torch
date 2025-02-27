
from hmc.model import ConstrainedFFNNModel, get_constr_out
from hmc.dataset import HMCDatasetManager, initialize_dataset
import os

import torch
import torch.utils.data
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from sklearn.metrics import average_precision_score


def train_global(dataset_name, args):
    print(".......................................")
    print("Experiment with {} dataset ".format(dataset_name))
    # Load train, val and test set
    data = dataset_name.split('_')[0]
    ontology = dataset_name.split('_')[1]
    hmc_dataset = initialize_dataset(dataset_name, args.dataset_path, output_path=args.output_path,
                                          is_global=True)
    train, val, test = hmc_dataset.get_torch_dataset()

    hmc_dataset.compute_matrix_R()
    R = hmc_dataset.R.to(args.device)

    train.X = torch.tensor(train.X).to(args.device)
    val.X = torch.tensor(val.X).to(args.device)

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
    model = ConstrainedFFNNModel(args.input_dims[data], args.hidden_dim, args.output_dims[ontology][data] + num_to_skip,
                                 args.hyperparams, R)
    model.to(args.device)

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
            x = x.to(args.device)
            labels = labels.to(args.device)

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
            x = x.to(args.device)
            y = y.to(args.device)

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