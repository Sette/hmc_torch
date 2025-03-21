import json
import os

import torch
from tensorflow.python.layers.core import dropout
from torch.utils.data import DataLoader
import torch.nn as nn

from hmc.model import HMCLocalClassificationModel

from hmc.model.losses import show_global_loss, show_local_losses
from hmc.utils.dir import create_job_id, create_dir
from hmc.model.arguments import get_parser
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from hmc.utils import create_dir
from hmc.dataset.manager import initialize_dataset_experiments


from sklearn.metrics import average_precision_score
from hmc.model.global_classifier import ConstrainedFFNNModel, get_constr_out
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import numpy as np


def train_local(dataset_name, args):
    print(".......................................")
    print("Experiment with {} dataset ".format(dataset_name))
    # Load train, val and test set
    device = torch.device(args.device)
    data, ontology = dataset_name.split('_')

    hmc_dataset = initialize_dataset_experiments(dataset_name, device=args.device, dataset_type='arff', is_global=False)
    train, valid, test = hmc_dataset.get_datasets()
    to_eval = torch.as_tensor(hmc_dataset.to_eval, dtype=torch.bool).clone().detach()
    dropouts = None
    thresholds = None
    experiment = True

    if experiment:
        args.hidden_dim= args.hidden_dims[ontology][data]
        args.lr = args.lrs[ontology][data]
        args.num_epochs = args.epochss[ontology][data]
        args.weight_decay =  1e-5
        args.batch_size = 4
        args.num_layers = 3
        args.dropout = 0.7
        args.non_lin = 'relu'

    args.hyperparams = {'batch_size': args.batch_size, 'num_layers': args.num_layers, 'dropout': args.dropout, 'non_lin': args.non_lin,
                   'hidden_dim': args.hidden_dim, 'lr': args.lr, 'weight_decay': args.weight_decay}

    scaler = preprocessing.StandardScaler().fit(np.concatenate((train.X, valid.X)))
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit(np.concatenate((train.X, valid.X)))
    valid.X, valid.Y = torch.tensor(scaler.transform(imp_mean.transform(valid.X))).clone().detach().to(
        device), torch.tensor(valid.Y).clone().detach().to(device)

    train.X, train.Y = torch.tensor(scaler.transform(imp_mean.transform(train.X))).clone().detach().to(
        device), torch.tensor(train.Y).clone().detach().to(device)
    test.X, test.Y = torch.as_tensor(scaler.transform(imp_mean.transform(test.X))).clone().detach().to(
        device), torch.as_tensor(test.Y).clone().detach().to(
        device)

    # Create loaders
    train_dataset = [(x, y) for (x, y) in zip(train.X, train.Y)]
    if ('others' not in args.datasets):
        val_dataset = [(x, y) for (x, y) in zip(valid.X, valid.Y)]
        for (x, y) in zip(valid.X, valid.Y):
            train_dataset.append((x, y))
    test_dataset = [(x, y) for (x, y) in zip(test.X, test.Y)]

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)

    num_epochs = args.num_epochs
    if 'GO' in dataset_name:
        num_to_skip = 4
    else:
        num_to_skip = 1


    params = {
        'levels_size': hmc_dataset.levels_size,
        'input_size': args.input_dims[data]
    }

    model = HMCLocalClassificationModel(**params)
    print(model)

    # Create the model
    # model = HMCLocalClassificationModel(levels_size=hmc_dataset.levels_size,
    #                                     input_size=args.input_dims[data],
    #                                     hidden_size=args.hidden_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

    criterions = [nn.BCEWithLogitsLoss(reduction='sum') for _ in hmc_dataset.levels_size]

    if torch.cuda.is_available():
        model = model.to(device)
        criterions = [criterion.to('cuda') for criterion in criterions]

    early_stopping_patience = 20
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        local_train_losses = [0.0 for _ in range(hmc_dataset.max_depth)]
        for inputs, targets in train_loader:
            if torch.cuda.is_available():
                inputs, targets = inputs.to('cuda'), [target.to('cuda') for target in targets]
            outputs = model(inputs.float())

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
        global_train_loss = sum(local_train_losses) / hmc_dataset.max_depth

        print(f'Epoch {epoch}/{args.epochs}')
        show_local_losses(local_train_losses, set='Train')
        show_global_loss(global_train_loss, set='Train')

        model.eval()
        local_val_losses = [0.0 for _ in range(hmc_dataset.max_depth)]
        with torch.no_grad():
            for inputs, targets in test_loader:
                if torch.cuda.is_available():
                    inputs, targets = inputs.to('cuda'), [target.to('cuda') for target in targets]
                outputs = model(inputs)

                total_val_loss = 0.0
                for index, (output, target) in enumerate(zip(outputs, targets)):
                    loss = criterions[index](output, target)
                    total_val_loss += loss
                    local_val_losses[index] += loss.item()

        local_val_losses = [loss / len(test_loader) for loss in local_val_losses]
        global_val_loss = sum(local_val_losses) / hmc_dataset.max_depth

        print(f'Epoch {epoch}/{args.epochs}')
        show_local_losses(local_val_losses, set='Val')
        show_global_loss(global_val_loss, set='Val')

        current_val_loss = round(global_val_loss, 4)
        if current_val_loss <= best_val_loss - 2e-4:
            best_val_loss = current_val_loss
            print('new best model')
            #torch.save(model.state_dict(), os.path.join(model_path, f'best_binary-{epoch}.pth'))
        else:
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered")
                return None

            patience_counter += 1
    return None



