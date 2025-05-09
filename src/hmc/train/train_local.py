import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from hmc.dataset.manager.dataset_manager import initialize_dataset_experiments
from hmc.model.local_classifier.baseline.model import HMCLocalModel
from hmc.model.losses import show_global_loss, show_local_losses
from hmc.train.utils import local_to_global_predictions
import optuna
import logging

logging.basicConfig(
    level=logging.INFO,  # Pode ser DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def run(args):
    if torch.cuda.is_available():
        args.model = args.model.to(args.device)
        args.criterions = [criterion.to(args.device) for criterion in args.criterions]

    args.early_stopping_patience = 3
    args.patience_counters = [0] * args.hmc_dataset.max_depth
    args.level_active = [True] * args.hmc_dataset.max_depth

    args.best_val_loss = [float("inf")] * args.max_depth
    logging.info(f"Best val loss created {args.best_val_loss}")

    args.epochs_to_eval = 10
    args.count_epochs_eval = 0

    optimizer = torch.optim.Adam(
        args.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    args.optimizer = optimizer

    for epoch in range(1, args.epochs + 1):
        args.model.train()
        local_train_losses = [0.0 for _ in range(args.hmc_dataset.max_depth)]
        args.active_levels = [i for i, active in enumerate(args.level_active) if active]
        logging.info(f"Active levels: {args.active_levels}")

        if not args.active_levels:
            logging.info("All levels have triggered early stopping.")
            break

        for inputs, targets, _ in args.train_loader:
            if torch.cuda.is_available():
                inputs, targets = inputs.to("cuda"), [
                    target.to("cuda") for target in targets
                ]
            outputs = args.model(inputs.float())

            # Zerar os gradientes antes de cada batch
            args.optimizer.zero_grad()

            for index in args.active_levels:
                output = outputs[index]
                target = targets[index].float()

                loss = args.criterions[index](output, target)
                local_train_losses[index] += loss

        # Backward pass (cálculo dos gradientes)
        for total_loss in local_train_losses:
            if total_loss > 0:
                total_loss.backward()
        args.optimizer.step()

        local_train_losses = [
            loss / len(args.train_loader) for loss in local_train_losses
        ]
        non_zero_losses = [loss for loss in local_train_losses if loss > 0]
        global_train_loss = (
            sum(non_zero_losses) / len(non_zero_losses) if non_zero_losses else 0
        )

        logging.info(f"Epoch {epoch}/{args.epochs}")
        show_local_losses(local_train_losses, set="Train")
        show_global_loss(global_train_loss, set="Train")

        local_val_losses, local_val_precision = val(args)
        logging.info(f"Local loss: {local_val_losses}")
        logging.info(f"Local precision: {local_val_precision}")
    return None


def val(args):
    args.model.eval()
    local_val_losses = [0.0] * args.max_depth
    output_val = [0.0] * args.max_depth
    y_val = [0.0] * args.max_depth
    local_val_precision = [0.0] * args.max_depth
    # if count_epochs_eval >= epochs_to_eval:
    #     count_epochs_eval = 0
    # Avaliar o modelo a cada epochs_to_eval épocas

    with torch.no_grad():
        for i, (inputs, targets, _) in enumerate(args.val_loader):
            if torch.cuda.is_available():
                inputs, targets = inputs.to("cuda"), [
                    target.to("cuda") for target in targets
                ]
            outputs = args.model(inputs.float())

            for index in args.active_levels:
                output = outputs[index]
                target = targets[index].float()
                loss = args.criterions[index](output, target)
                local_val_losses[index] += loss

                if i == 0:
                    output_val[index] = output.to("cpu")
                    y_val[index] = target.to("cpu")
                else:
                    output_val[index] = torch.cat(
                        (output_val[index], output.to("cpu")), dim=0
                    )
                    y_val[index] = torch.cat((y_val[index], target.to("cpu")), dim=0)
    for idx in range(args.max_depth):
        local_val_precision[idx] = average_precision_score(
            y_val[idx], output_val[idx], average="micro"
        )

    local_val_losses = [loss / len(args.val_loader) for loss in local_val_losses]
    logging.info(f"Levels to evaluate: {args.active_levels}")
    for i in args.active_levels:
        if round(local_val_losses[i].item(), 3) < round(args.best_val_loss[i], 3):
            args.best_val_loss[i] = round(local_val_losses[i].item(), 3)
            args.patience_counters[i] = 0
            logging.info(f"Level {i}: improved (loss={local_val_losses[i]:.4f})")
        else:
            args.patience_counters[i] += 1
            logging.info(
                f"Level {i}: no improvement \
                (patience {args.patience_counters[i]}/{args.early_stopping_patience})"
            )
            if args.patience_counters[i] >= args.early_stopping_patience:
                args.level_active[i] = False
                logging.info(
                    f"🚫 Early stopping triggered for level {i} — freezing its parameters"
                )
                # ❄️ Congelar os parâmetros desse nível
                for param in args.model.levels[i].parameters():
                    param.requires_grad = False
    return local_val_losses, local_val_precision


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
    logging.info(f"Local test score: {local_val_score}")
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

    score = average_precision_score(
        Y_true_global_original[:, to_eval], Y_pred_global[:, to_eval], average="micro"
    )

    logging.info(score)

    return None


def train_local(args):
    logging.info(".......................................")
    logging.info("Experiment with {} dataset ".format(args.dataset_name))
    # Load train, val and test set
    device = torch.device(args.device)
    data, ontology = args.dataset_name.split("_")
    hmc_dataset = initialize_dataset_experiments(
        args.dataset_name,
        device=args.device,
        dataset_type="arff",
        is_global=False,
    )
    data_train, data_valid, data_test = hmc_dataset.get_datasets()
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
        args.n_trials = 20
        args.data = data

    scaler = preprocessing.StandardScaler().fit(
        np.concatenate((data_train.X, data_valid.X))
    )
    imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean").fit(
        np.concatenate((data_train.X, data_valid.X))
    )
    data_valid.X = (
        torch.tensor(scaler.transform(imp_mean.transform(data_valid.X)))
        .clone()
        .detach()
        .to(device)
    )
    data_train.X = (
        torch.tensor(scaler.transform(imp_mean.transform(data_train.X)))
        .clone()
        .detach()
        .to(device)
    )
    data_test.X = (
        torch.as_tensor(scaler.transform(imp_mean.transform(data_test.X)))
        .clone()
        .detach()
        .to(device)
    )

    data_test.Y = torch.as_tensor(data_test.Y).clone().detach().to(device)
    data_valid.Y = torch.tensor(data_valid.Y).clone().detach().to(device)
    data_train.Y = torch.tensor(data_train.Y).clone().detach().to(device)

    # Create loaders using local (per-level) y labels
    train_dataset = [
        (x, y_levels, y)
        for (x, y_levels, y) in zip(data_train.X, data_train.Y_local, data_train.Y)
    ]

    val_dataset = [
        (x, y_levels, y)
        for (x, y_levels, y) in zip(data_valid.X, data_valid.Y_local, data_valid.Y)
    ]

    test_dataset = [
        (x, y_levels, y)
        for (x, y_levels, y) in zip(data_test.X, data_test.Y_local, data_test.Y)
    ]

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=False
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=args.batch_size, shuffle=False
    )
    args.train_loader = train_loader
    args.val_loader = val_loader
    args.test_loader = test_loader
    args.hmc_dataset = hmc_dataset
    args.levels_size = hmc_dataset.levels_size
    args.input_dim = args.input_dims[data]
    args.max_depth = hmc_dataset.max_depth
    args.to_eval = hmc_dataset.to_eval

    criterions = [nn.BCELoss() for _ in hmc_dataset.levels_size]
    args.criterions = criterions

    if args.hpo == "true":
        logging.info("Hyperparameter optimization")
        best_params = optimize_hyperparameters_per_level(args=args)

        logging.info(best_params)
    else:

        params = {
            "levels_size": hmc_dataset.levels_size,
            "input_size": args.input_dims[data],
            "hidden_size": [args.hidden_dim] * len(hmc_dataset.levels_size),
            "num_layers": [args.num_layers] * len(hmc_dataset.levels_size),
            "dropout": [args.dropout] * len(hmc_dataset.levels_size),
        }

        model = HMCLocalModel(**params)
        args.model = model
        logging.info(model)
        # Create the model
        # model = HMCLocalClassificationModel(levels_size=hmc_dataset.levels_size,
        #                                     input_size=args.input_dims[data],
        #                                     hidden_size=args.hidden_dim)
        run(args)


def optimize_hyperparameters_per_level(args):
    def objective(trial, level):
        hidden_dim = [
            trial.suggest_categorical("hidden_dim", [64, 128, 256])
            for _ in range(args.max_depth)
        ]
        # lr = [trial.suggest_float("lr", 1e-4, 1e-2, log=True) for _ in range(args.max_depth)]
        lr_by_level = {
            i: trial.suggest_float(f"lr_level_{i}", 1e-4, 1e-2, log=True)
            for i in range(args.max_depth)
        }
        dropout = [
            trial.suggest_float("dropout", 0.3, 0.8) for _ in range(args.max_depth)
        ]
        num_layers = [
            trial.suggest_int("num_layers", 1, 3) for _ in range(args.max_depth)
        ]
        weight_decay = [
            trial.suggest_float("lr", 1e-6, 1e-2, log=True)
            for _ in range(args.max_depth)
        ]

        params = {
            "levels_size": args.levels_size,
            "input_size": args.input_dims[args.data],
            "hidden_size": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
        }

        args.model = HMCLocalModel(**params).to(args.device)

        optimizer = torch.optim.Adam(
            args.model.parameters(),
            lr=lr_by_level[level],
            weight_decay=weight_decay[level],
        )
        args.optimizer = optimizer

        if torch.cuda.is_available():
            args.model = args.model.to(args.device)
            args.criterions = [
                criterion.to(args.device) for criterion in args.criterions
            ]

        args.early_stopping_patience = 3
        args.patience_counters = [0] * args.hmc_dataset.max_depth
        args.level_active = [True] * args.hmc_dataset.max_depth

        args.best_val_loss = [float("inf")] * args.max_depth
        logging.info(f"Best val loss created {args.best_val_loss}")

        args.epochs_to_eval = 10
        args.count_epochs_eval = 0

        for epoch in range(1, args.epochs + 1):
            args.model.train()
            local_train_losses = [0.0 for _ in range(args.hmc_dataset.max_depth)]
            args.active_levels = [
                i for i, active in enumerate(args.level_active) if active
            ]
            logging.info(f"Active levels: {args.active_levels}")

            if not args.active_levels:
                logging.info("All levels have triggered early stopping.")
                break

            for inputs, targets, _ in args.train_loader:
                if torch.cuda.is_available():
                    inputs, targets = inputs.to("cuda"), [
                        target.to("cuda") for target in targets
                    ]
                outputs = args.model(inputs.float())

                # Zerar os gradientes antes de cada batch
                args.optimizer.zero_grad()

                for index in args.active_levels:
                    output = outputs[index]
                    target = targets[index].float()

                    loss = args.criterions[index](output, target)
                    local_train_losses[index] += loss

            # Backward pass (cálculo dos gradientes)
            for total_loss in local_train_losses:
                if total_loss > 0:
                    total_loss.backward()
            args.optimizer.step()

            local_train_losses = [
                loss / len(args.train_loader) for loss in local_train_losses
            ]
            non_zero_losses = [loss for loss in local_train_losses if loss > 0]
            global_train_loss = (
                sum(non_zero_losses) / len(non_zero_losses) if non_zero_losses else 0
            )

            logging.info(f"Epoch {epoch}/{args.epochs}")
            show_local_losses(local_train_losses, set="Train")
            show_global_loss(global_train_loss, set="Train")

            local_val_losses, local_val_precision = val(args)
            logging.info(f"Local loss: {local_val_losses}")
            logging.info(f"Local precision: {local_val_precision}")

            return sum(local_val_precision) / args.max_depth

    best_params_per_level = {}

    for level in range(len(args.levels_size)):
        logging.info(f"\n🔍 Optimizing hyperparameters for level {level}...\n")
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, level), n_trials=args.n_trials)
        best_params_per_level[level] = study.best_params
        logging.info(f"✅ Best hyperparameters for level {level}: {study.best_params}")

    return best_params_per_level
