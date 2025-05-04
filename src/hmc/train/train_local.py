import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from hmc.dataset.manager.dataset_manager import initialize_dataset_experiments
from hmc.model.local_classifier.baseline.model import HMCLocalClassificationModel
from hmc.model.losses import show_global_loss, show_local_losses
import optuna

# Converter local -> global


def local_to_global_predictions(local_labels, local_nodes_idx, nodes_idx):
    n_samples = local_labels[0].shape[0]
    n_global_labels = len(nodes_idx)
    global_preds = np.zeros((n_samples, n_global_labels))
    sorted_levels = sorted(local_nodes_idx.keys())
    local_nodes_reverse = {level: {v: k for k, v in local_nodes_idx[level].items()} for level in sorted_levels}
    # print(f"Local nodes idx: {local_nodes_idx}")
    # print(f"Local nodes reverse: {local_nodes_reverse}")

    print(f"Exemplos: {n_samples}")
    # print(f"Shape local_preds: {len(local_labels)}")
    # print(f"Local nodes idx: {local_nodes_reverse}")

    # Etapa 1: montar node_names ativados por exemplo
    activated_nodes_by_example = [[] for _ in range(n_samples)]

    for level_index, level in enumerate(sorted_levels):
        level_preds = local_labels[level_index]  # shape: [n_samples, n_classes_at_level]
        for idx_example, label in enumerate(level_preds):
            local_indices = np.where(label == 1)[0]  # aceita floats ou binários
            for local_idx in local_indices:
                node_name = local_nodes_reverse[level].get(local_idx)
                if node_name:
                    activated_nodes_by_example[idx_example].append(node_name)
                else:
                    print(f"[WARN] Índice local {local_idx} não encontrado no nível {level}")

    # print(f"Node names ativados por exemplo: {activated_nodes_by_example[0]}")
    global_indices = []
    for node in activated_nodes_by_example[0]:
        # print(f"Node names ativados: {node}")
        if "/" in node:
            node = node.replace("/", ".")
        global_indices.append(nodes_idx.get(node))
    print(global_indices)
    # Etapa 2: converter node_names para índices globais
    for idx_example, node_names in enumerate(activated_nodes_by_example):
        for node_name in node_names:
            key = node_name.replace("/", ".")
            if key in nodes_idx:
                global_idx = nodes_idx[key]
                global_preds[idx_example][global_idx] = 1
            else:
                print(f"[WARN] Node '{key}' não encontrado em nodes_idx")

    return global_preds


def run(
    model,
    hmc_dataset,
    train_loader,
    val_loader,
    criterions,
    optimizer,
    device,
    epochs=20,
):
    """
    Train a hierarchical multi-class (HMC) model with early stopping and evaluation.

    Args:
        model (torch.nn.Module): The hierarchical multi-class model to be trained.
        hmc_dataset (Dataset): The dataset containing hierarchical multi-class data.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        criterions (list of torch.nn.Module): List of loss functions,
        one for each level of the hierarchy.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        device (torch.device): Device to run the training on (e.g., 'cuda' or 'cpu').
        epochs (int, optional): Number of epochs to train the model. Defaults to 20.

    Notes:
        - Implements early stopping for each level of the hierarchy based on validation loss.
        - Freezes parameters of levels that trigger early stopping.
        - Evaluates the model every `epochs_to_eval` epochs and
            calculates local and global scores.
        - Outputs training and validation losses,
            as well as evaluation metrics, during training.
    """
    if torch.cuda.is_available():
        model = model.to(device)
        criterions = [criterion.to(device) for criterion in criterions]
    early_stopping_patience = 3
    patience_counters = [0] * hmc_dataset.max_depth
    level_active = [True] * hmc_dataset.max_depth

    epochs_to_eval = 10
    count_epochs_eval = 0

    for epoch in range(1, epochs + 1):
        model.train()
        local_train_losses = [0.0 for _ in range(hmc_dataset.max_depth)]
        active_levels = [i for i, active in enumerate(level_active) if active]

        if not active_levels:
            print("All levels have triggered early stopping.")
            break

        for inputs, targets, _ in train_loader:
            if torch.cuda.is_available():
                inputs, targets = inputs.to("cuda"), [target.to("cuda") for target in targets]
            outputs = model(inputs.float())

            # Zerar os gradientes antes de cada batch
            optimizer.zero_grad()

            for index in active_levels:
                output = outputs[index]
                target = targets[index].float()

                loss = criterions[index](output, target)
                local_train_losses[index] += loss

        # Backward pass (cálculo dos gradientes)
        for total_loss in local_train_losses:
            if total_loss > 0:
                total_loss.backward()
        optimizer.step()

        local_train_losses = [loss / len(train_loader) for loss in local_train_losses]
        non_zero_losses = [loss for loss in local_train_losses if loss > 0]
        global_train_loss = sum(non_zero_losses) / len(non_zero_losses) if non_zero_losses else 0

        print(f"Epoch {epoch}/{epochs}")
        show_local_losses(local_train_losses, set="Train")
        show_global_loss(global_train_loss, set="Train")

        val(
            count_epochs_eval,
            epochs_to_eval,
            model,
            active_levels,
            level_active.best_val_loss,
            patience_counters,
            early_stopping_patience,
            hmc_dataset,
            val_loader,
            criterions,
        )

    return None


def optimize_hyperparameters_per_level(
    train_loader,
    val_loader,
    args,
):
    def objective(trial, level):
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.3, 0.8)
        num_layers = trial.suggest_int("num_layers", 1, 3)

        params = {
            "levels_size": args.levels_size,
            "input_size": args.input_dims[args.data],
            "hidden_size": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
        }

        model = HMCLocalClassificationModel(**params).to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

        criterions = [nn.BCELoss() for _ in args.levels_size]

        early_stopping_patience = 3
        patience_counters = [0] * args.max_depth
        level_active = [True] * args.max_depth

        epochs_to_eval = 10
        count_epochs_eval = 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            local_train_losses = [0.0 for _ in range(args.max_depth)]
            active_levels = [i for i, active in enumerate(level_active) if active]

            if not active_levels:
                print("All levels have triggered early stopping.")
                break

            for inputs, targets, _ in train_loader:
                if torch.cuda.is_available():
                    inputs, targets = inputs.to("cuda"), [target.to("cuda") for target in targets]
                outputs = model(inputs.float())

                # Zerar os gradientes antes de cada batch
                optimizer.zero_grad()

                for index in active_levels:
                    output = outputs[index]
                    target = targets[index].float()

                    loss = criterions[index](output, target)
                    local_train_losses[index] += loss

            # Backward pass (cálculo dos gradientes)
            for total_loss in local_train_losses:
                if total_loss > 0:
                    total_loss.backward()
            optimizer.step()

            local_train_losses = [loss / len(train_loader) for loss in local_train_losses]
            non_zero_losses = [loss for loss in local_train_losses if loss > 0]
            global_train_loss = sum(non_zero_losses) / len(non_zero_losses) if non_zero_losses else 0

            print(f"Epoch {epoch}/{args.epochs}")
            show_local_losses(local_train_losses, set="Train")
            show_global_loss(global_train_loss, set="Train")

            local_val_losses = val(
                count_epochs_eval,
                epochs_to_eval,
                model,
                active_levels,
                level_active,
                patience_counters,
                early_stopping_patience,
                val_loader,
                criterions,
                args,
            )

            return sum(local_val_losses) / len(local_val_losses)

    best_params_per_level = {}

    for level in range(len(args.levels_size)):
        print(f"\n🔍 Optimizing hyperparameters for level {level}...\n")
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, level), n_trials=args.n_trials)
        best_params_per_level[level] = study.best_params
        print(f"✅ Best hyperparameters for level {level}: {study.best_params}")

    return best_params_per_level


def val(
    count_epochs_eval,
    epochs_to_eval,
    model,
    active_levels,
    level_active,
    patience_counters,
    early_stopping_patience,
    val_loader,
    criterions,
    args,
):
    model.eval()
    count_epochs_eval += 1
    local_val_losses = [0.0 for _ in range(args.max_depth)]
    best_val_loss = [float("inf")] * args.max_depth
    # if count_epochs_eval >= epochs_to_eval:
    #     count_epochs_eval = 0
    # Avaliar o modelo a cada epochs_to_eval épocas
    model.eval()
    with torch.no_grad():
        for inputs, targets, _ in val_loader:
            if torch.cuda.is_available():
                inputs, targets = inputs.to("cuda"), [target.to("cuda") for target in targets]
            outputs = model(inputs.float())

            for index in active_levels:
                output = outputs[index]
                target = targets[index].float()

                loss = criterions[index](output, target)
                local_val_losses[index] += loss.item()

    local_val_losses = [loss / len(val_loader) for loss in local_val_losses]

    for i in active_levels:
        if round(local_val_losses[i], 3) < round(best_val_loss[i], 3):
            best_val_loss[i] = round(local_val_losses[i], 3)
            patience_counters[i] = 0
            print(f"Level {i}: improved (loss={local_val_losses[i]:.4f})")
        else:
            patience_counters[i] += 1
            print(
                f"Level {i}: no improvement \
                (patience {patience_counters[i]}/{early_stopping_patience})"
            )
            if patience_counters[i] >= early_stopping_patience:
                level_active[i] = False
                print(f"🚫 Early stopping triggered for level {i} — freezing its parameters")
                # ❄️ Congelar os parâmetros desse nível
                for param in model.levels[i].parameters():
                    param.requires_grad = False
        return local_val_losses


def test(
    model,
    hmc_dataset,
    test_loader,
):
    model.eval()
    local_inputs = [[] for _ in range(hmc_dataset.max_depth)]
    local_outputs = [[] for _ in range(hmc_dataset.max_depth)]
    to_eval = torch.as_tensor(hmc_dataset.to_eval, dtype=torch.bool).clone().detach()
    Y_true_global = []
    with torch.no_grad():
        for inputs, targets, global_targets in test_loader:
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
                targets = [target.to("cuda").float() for target in targets]
                global_targets = global_targets.to("cpu")
            outputs = model(inputs.float())

            for index, (output, target) in enumerate(zip(outputs, targets)):
                output = output.to("cpu")
                target = target.to("cpu")
                local_inputs[index].append(target)
                local_outputs[index].append(output)
            Y_true_global.append(global_targets)
        # Concat all outputs and targets by level
    local_inputs = [torch.cat(targets, dim=0) for targets in local_inputs]
    local_outputs = [torch.cat(outputs, dim=0) for outputs in local_outputs]

    # Get local scores
    local_val_score = []
    for target, output in zip(local_inputs, local_outputs):
        score = average_precision_score(target, output, average="micro")  # micro score
        local_val_score.append(score)
    print(f"Local test score: {local_val_score}")
    # Concat global targets
    Y_true_global_original = torch.cat(Y_true_global, dim=0).numpy()

    Y_pred_global = local_to_global_predictions(
        local_outputs,
        hmc_dataset.train.local_nodes_idx,
        hmc_dataset.train.nodes_idx,
    )

    # Y_true_global_converted = local_to_global_predictions(
    #     local_inputs, train.local_nodes_idx, train.nodes_idx
    # )

    score = average_precision_score(Y_true_global_original[:, to_eval], Y_pred_global[:, to_eval], average="micro")

    print(score)

    return None


def train_local(dataset_name, args):
    print(".......................................")
    print("Experiment with {} dataset ".format(dataset_name))
    # Load train, val and test set
    device = torch.device(args.device)
    data, ontology = dataset_name.split("_")

    hmc_dataset = initialize_dataset_experiments(dataset_name, device=args.device, dataset_type="arff", is_global=False)
    train, valid, test = hmc_dataset.get_datasets()
    experiment = True

    if experiment:
        args.hidden_dim = args.hidden_dims[ontology][data]
        args.lr = args.lrs[ontology][data]
        # args.num_epochs = args.epochss[ontology][data]
        args.weight_decay = 1e-5
        args.batch_size = 4
        args.num_layers = 3
        args.dropout = 0.7
        args.non_lin = "relu"
        args.epochs = 20
        args.n_trials = 20
        args.data = data
        args.levels_size = hmc_dataset.levels_size
        args.input_dim = args.input_dims[data]
        args.max_depth = hmc_dataset.max_depth
        args.to_eval = hmc_dataset.to_eval

    scaler = preprocessing.StandardScaler().fit(np.concatenate((train.X, valid.X)))
    imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean").fit(np.concatenate((train.X, valid.X)))
    valid.X = torch.tensor(scaler.transform(imp_mean.transform(valid.X))).clone().detach().to(device)
    train.X = torch.tensor(scaler.transform(imp_mean.transform(train.X))).clone().detach().to(device)
    test.X = torch.as_tensor(scaler.transform(imp_mean.transform(test.X))).clone().detach().to(device)

    test.Y = torch.as_tensor(test.Y).clone().detach().to(device)
    valid.Y = torch.tensor(valid.Y).clone().detach().to(device)
    train.Y = torch.tensor(train.Y).clone().detach().to(device)

    # Create loaders using local (per-level) y labels
    train_dataset = [(x, y_levels, y) for (x, y_levels, y) in zip(train.X, train.Y_local, train.Y)]

    val_dataset = [(x, y_levels, y) for (x, y_levels, y) in zip(valid.X, valid.Y_local, valid.Y)]

    # test_dataset = [(x, y_levels, y) for (x, y_levels, y) in zip(test.X, test.Y_local, test.Y)]

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)

    best_params = optimize_hyperparameters_per_level(
        train_loader=train_loader,
        val_loader=val_loader,
        args=args,
    )

    print(best_params)

    # params = {
    #     "levels_size": hmc_dataset.levels_size,
    #     "input_size": args.input_dims[data],
    # }

    # model = HMCLocalClassificationModel(**params)
    # print(model)
    # # Create the model
    # # model = HMCLocalClassificationModel(levels_size=hmc_dataset.levels_size,
    # #                                     input_size=args.input_dims[data],
    # #                                     hidden_size=args.hidden_dim)

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # criterions = [nn.BCELoss() for _ in hmc_dataset.levels_size]

    # run(
    #     model,
    #     hmc_dataset,
    #     train_loader,
    #     test_loader,
    #     criterions,
    #     optimizer,
    #     device,
    #     args.num_epochs,
    # )
