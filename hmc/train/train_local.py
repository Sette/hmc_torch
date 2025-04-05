from hmc.model import HMCLocalClassificationModel

from hmc.model.losses import show_global_loss, show_local_losses
from hmc.dataset.manager import initialize_dataset_experiments

from sklearn.metrics import average_precision_score
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import numpy as np


# Converter local -> global
def local_to_global_predictions(local_preds, local_nodes_idx, nodes_idx):
    n_samples = local_preds[0].shape[0]
    n_global_labels = len(nodes_idx)
    global_preds = np.zeros((n_samples, n_global_labels))
    local_nodes_reverse = [{v: k for k, v in local_nodes.items()} for local_nodes in local_nodes_idx.values()]

    print(f'Exemplos: {n_samples}')
    print(f'Shape local_preds: {len(local_preds)}')
    print(f'Local nodes idx: {local_nodes_reverse}')

    for index, local_labels in enumerate(local_preds):
        for idx_example, local_label in enumerate(local_labels):
            print(np.where(local_label == 1)[0])
            local_indices = [i for i, x in enumerate(local_labels[index]) if x == 1]
            print(type(local_indices))
            for local_indice in local_indices:
                node_name = local_nodes_reverse[index].get(local_indice)
                global_idx = nodes_idx[node_name.replace('/', '.')]
                global_preds[idx_example][:, global_idx] = 1




    # for level, label_to_local_idx in local_nodes_idx.items():
    #     for node_name, local_idx in label_to_local_idx.items():
    #         global_idx = nodes_idx[node_name.replace('/', '.')]
    #         global_preds[:, global_idx] = local_preds[level][:, local_idx]

    return global_preds

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
        #args.num_epochs = args.epochss[ontology][data]
        args.weight_decay =  1e-5
        args.batch_size = 4
        args.num_layers = 3
        args.dropout = 0.7
        args.non_lin = 'relu'

    args.hyperparams = {'batch_size': args.batch_size, 'num_layers': args.num_layers, 'dropout': args.dropout, 'non_lin': args.non_lin,
                   'hidden_dim': args.hidden_dim, 'lr': args.lr, 'weight_decay': args.weight_decay}

    scaler = preprocessing.StandardScaler().fit(np.concatenate((train.X, valid.X)))
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit(np.concatenate((train.X, valid.X)))
    valid.X = torch.tensor(scaler.transform(imp_mean.transform(valid.X))).clone().detach().to(
        device)

    train.X = torch.tensor(scaler.transform(imp_mean.transform(train.X))).clone().detach().to(device)
    test.X = torch.as_tensor(scaler.transform(imp_mean.transform(test.X))).clone().detach().to(device)

    test.Y = torch.as_tensor(test.Y).clone().detach().to(device)
    valid.Y = torch.tensor(valid.Y).clone().detach().to(device)
    train.Y =  torch.tensor(train.Y).clone().detach().to(device)

    # Create loaders using local (per-level) y labels
    train_dataset = [(x, y_levels, y) for (x, y_levels, y) in zip(train.X, train.Y_local, train.Y)]

    # Optionally extend train with validation
    if ('others' not in args.datasets):
        val_dataset = [(x, y_levels, y) for (x, y_levels, y) in zip(valid.X, valid.Y_local, valid.Y)]
        train_dataset.extend(val_dataset)

    test_dataset = [(x, y_levels, y) for (x, y_levels, y) in zip(test.X, test.Y_local, train.Y)]

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)

   
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

    criterions = [nn.BCELoss() for _ in hmc_dataset.levels_size]

    if torch.cuda.is_available():
        model = model.to(device)
        criterions = [criterion.to('cuda') for criterion in criterions]

    early_stopping_patience = 5
    best_train_losses = [float('inf')] * hmc_dataset.max_depth
    patience_counters = [0] * hmc_dataset.max_depth
    level_active = [True] * hmc_dataset.max_depth


    for epoch in range(1, args.num_epochs + 1):
        model.train()
        local_train_losses = [0.0 for _ in range(hmc_dataset.max_depth)]
        active_levels = [i for i, active in enumerate(level_active) if active]

        if not active_levels:
            print("All levels have triggered early stopping.")
            break

        for inputs, targets, _ in train_loader:
            if torch.cuda.is_available():
                inputs, targets = inputs.to('cuda'), [target.to('cuda') for target in targets]
            outputs = model(inputs.float())

            # Zerar os gradientes antes de cada batch
            optimizer.zero_grad()

            total_loss = 0.0
            for index in active_levels:
                output = outputs[index]
                target = targets[index].float()
                # Máscara para exemplos com pelo menos uma label positiva ou negativa
                mask = target.sum(dim=1) != 0  # shape: [batch_size]
                if mask.sum() == 0:
                    continue  # Nenhum exemplo válido nesse nível
                
                # Aplicar máscara a output e target
                filtered_output = output[mask]
                filtered_target = target[mask]
                
                loss = criterions[index](filtered_output, filtered_target)
                local_train_losses[index] += loss

        
        # Backward pass (cálculo dos gradientes)
        [total_loss.backward() for total_loss in local_train_losses if total_loss > 0]
        optimizer.step()

        local_train_losses = [loss / len(train_loader) for loss in local_train_losses]
        global_train_loss = sum(local_train_losses) / hmc_dataset.max_depth
        

        print(f'\nEpoch {epoch}/{args.num_epochs}')

        for i in active_levels:
            if round(local_train_losses[i].item(), 3) < round(best_train_losses[i], 3):
                best_train_losses[i] = round(local_train_losses[i].item(), 3)
                patience_counters[i] = 0
                print(f"Level {i}: improved (loss={local_train_losses[i].item():.4f})")
            else:
                patience_counters[i] += 1
                print(f"Level {i}: no improvement (patience {patience_counters[i]}/{early_stopping_patience})")
                if patience_counters[i] >= early_stopping_patience:
                    level_active[i] = False
                    print(f"🚫 Early stopping triggered for level {i} — freezing its parameters")
                    # ❄️ Congelar os parâmetros desse nível
                    for param in model.levels[i].parameters():
                        param.requires_grad = False

        print(f'Epoch {epoch}/{args.num_epochs}')
        show_local_losses(local_train_losses, set='Train')
        show_global_loss(global_train_loss, set='Train')

    model.eval()
    local_val_losses = [0.0 for _ in range(hmc_dataset.max_depth)]
    local_inputs = [[] for _ in range(hmc_dataset.max_depth)]
    local_outputs = [[] for _ in range(hmc_dataset.max_depth)]
    Y_true_global = []
    with (torch.no_grad()):
        for inputs, targets, global_targets  in test_loader:
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')
                targets = [target.to('cuda').float() for target in targets]
                global_targets = global_targets.to('cpu')
            outputs = model(inputs.float())

            total_val_loss = 0.0
            for index, (output, target) in enumerate(zip(outputs, targets)):
                loss = criterions[index](output, target)
                total_val_loss += loss
                local_val_losses[index] += loss.item()
                output = output.to('cpu')
                target = target.to('cpu')
                local_inputs[index].append(target)
                local_outputs[index].append(output)
            Y_true_global.append(global_targets)
        # Concat all outputs and targets by level
    local_inputs = [torch.cat(targets, dim=0) for targets in local_inputs]
    local_outputs = [torch.cat(outputs, dim=0) for outputs in local_outputs]

    # Get local scores
    local_val_score = [
        average_precision_score(target, output, average='micro')
        for target, output in zip(local_inputs, local_outputs)
    ]
    print(f'Local test score: {local_val_score}')
    # Concat global targets
    #Y_true_global_original = torch.cat(Y_true_global, dim=0).numpy()

    
    #Y_pred_global = local_to_global_predictions(local_outputs, train.local_nodes_idx, train.nodes_idx)
    #Y_true_global_convertida = local_to_global_predictions(local_inputs, train.local_nodes_idx, train.nodes_idx)

    #print(f'Labels convertidas: {Y_true_global_convertida[0]}')
    #print(f'Labels verdadeiras: {Y_true_global_original[0]}')
    #print(f'Labels locais: {local_inputs[1][0]}')
    #print(f'Shape Y_true global: {Y_true_global.shape}')
    #(f'Shape Y_pred global convertido: {Y_pred_global.shape}')
    # Score global
    #global_score = average_precision_score(Y_true_global[:, to_eval], Y_pred_global[:, to_eval], average='micro')

    #local_val_losses = [loss / len(test_loader) for loss in local_val_losses]
    #global_val_loss = sum(local_val_losses) / hmc_dataset.max_depth


    #print(f'Global test score: {global_score}')
    #print(f'Global test loss: {global_val_loss}')
    # score = average_precision_score(y_test[:, to_eval], outputs, average='micro')
    #local_val_losses = [loss / len(test_loader) for loss in local_val_losses]
    #global_val_loss = sum(local_val_losses) / hmc_dataset.max_depth

    #print(f'Global test loss:{global_val_loss}')

    return None



