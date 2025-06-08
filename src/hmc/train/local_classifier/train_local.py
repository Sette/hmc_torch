import json
import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from hmc.dataset.manager.dataset_manager import initialize_dataset_experiments
from hmc.model.local_classifier.baseline.model import HMCLocalModel
from hmc.train.local_classifier.hpo.hpo_local import optimize_hyperparameters_per_level
from hmc.train.utils import (
    local_to_global_predictions,
    show_global_loss,
    show_local_losses,
    show_local_precision,
    create_job_id_name,
)
from hmc.utils.dir import create_dir


def save_dict_to_json(dictionary, file_path):
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(dictionary, json_file, ensure_ascii=False, indent=4)


def run(args):
    args.model = args.model.to(args.device)
    args.criterions = [criterion.to(args.device) for criterion in args.criterions]

    args.early_stopping_patience = 3
    args.patience_counters = [0] * args.hmc_dataset.max_depth
    args.level_active = [True] * args.hmc_dataset.max_depth

    args.best_val_loss = [float("inf")] * args.max_depth
    logging.info(f"Best val loss created {args.best_val_loss}")

    # optimizer = torch.optim.Adam(
    #     args.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    # )
    # args.optimizer = optimizer

    optimizers = [
        torch.optim.Adam(
            model.parameters(),
            lr=args.lr_values[idx],
            weight_decay=args.weight_decay_values[idx],
        )
        for idx, model in enumerate(args.model.levels)
    ]
    args.optimizers = optimizers

    for epoch in range(1, args.epochs + 1):
        args.model.train()
        local_train_losses = [0.0 for _ in range(args.hmc_dataset.max_depth)]
        # args.active_levels = [i for i, active in enumerate(args.level_active) if active]
        logging.info(f"Active levels: {args.active_levels}")

        if not args.active_levels:
            logging.info("All levels have triggered early stopping.")
            break

        for inputs, targets, _ in args.train_loader:

            inputs, targets = inputs.to(args.device), [
                target.to(args.device) for target in targets
            ]
            outputs = args.model(inputs.float())

            # Zerar os gradientes antes de cada batch
            # args.optimizer.zero_grad()
            for optimizer in args.optimizers:
                optimizer.zero_grad()

            for index in args.active_levels:
                output = outputs[index]
                target = targets[index].float()

                loss = args.criterions[index](output, target)
                local_train_losses[index] += loss

        # Backward pass (cálculo dos gradientes)
        for total_loss in local_train_losses:
            if total_loss > 0:
                total_loss.backward()
        # args.optimizer.step()
        for optimizer in args.optimizers:
            optimizer.step()

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

        if epoch % args.epochs_to_evaluate == 0:
            local_val_losses, local_val_precision = val(args)
            show_local_losses(local_val_losses, set="Val")
            show_local_precision(local_val_precision, set="Val")
    return None


def val(args):
    args.model.eval()
    local_val_losses = [0.0] * args.max_depth
    output_val = [0.0] * args.max_depth
    y_val = [0.0] * args.max_depth
    local_val_precision = [0.0] * args.max_depth

    with torch.no_grad():
        for i, (inputs, targets, _) in enumerate(args.val_loader):
            inputs, targets = inputs.to(args.device), [
                target.to(args.device) for target in targets
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
    for idx in args.active_levels:
        local_val_precision[idx] = average_precision_score(
            y_val[idx], output_val[idx], average="micro"
        )

    local_val_losses = [loss / len(args.val_loader) for loss in local_val_losses]
    logging.info(f"Levels to evaluate: {args.active_levels}")
    for i in args.active_levels:
        if round(local_val_losses[i].item(), 4) < args.best_val_loss[i]:
            args.best_val_loss[i] = round(local_val_losses[i].item(), 4)
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
                args.active_levels.remove(i)
                logging.info(
                    f"🚫 Early stopping triggered for level {i} — freezing its parameters"
                )
                # ❄️ Congelar os parâmetros desse nível
                for param in args.model.levels[i].parameters():
                    param.requires_grad = False
    return local_val_losses, local_val_precision


def test(args):
    args.model.eval()
    local_inputs = [[] for _ in range(args.hmc_dataset.max_depth)]
    local_outputs = [[] for _ in range(args.hmc_dataset.max_depth)]
    # to_eval = (
    #     torch.as_tensor(args.hmc_dataset.to_eval, dtype=torch.bool).clone().detach()
    # )
    Y_true_global = []
    with torch.no_grad():
        for inputs, targets, global_targets in args.test_loader:

            inputs = inputs.to(args.device)
            targets = [target.to(args.device).float() for target in targets]
            global_targets = global_targets.to("cpu")
            outputs = args.model(inputs.float())

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
    logging.info("Local test score: %s", str(local_val_score))

    job_id = create_job_id_name(prefix="test")

    create_dir("results/train")

    save_dict_to_json(
        score,
        f"results/train/{args.dataset_name}-{job_id}.json",
    )

    # Concat global targets
    # Y_true_global_original = torch.cat(Y_true_global, dim=0).numpy()

    # Y_pred_global = local_to_global_predictions(
    #     local_outputs,
    #     args.hmc_dataset.train.local_nodes_idx,
    #     args.hmc_dataset.train.nodes_idx,
    # )

    # Y_true_global_converted = local_to_global_predictions(
    #     local_inputs, train.local_nodes_idx, train.nodes_idx
    # )

    # score = average_precision_score(
    #     Y_true_global_original[:, to_eval], Y_pred_global[:, to_eval], average="micro"
    # )

    # logging.info(score)

    return None


def train_local(args):
    logging.info(".......................................")
    logging.info("Experiment with {} dataset ".format(args.dataset_name))
    # Load train, val and test set

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA não está disponível. Usando CPU.")
        args.device = torch.device("cpu")
    else:
        args.device = torch.device(args.device)

    args.data, args.ontology = args.dataset_name.split("_")
    hmc_dataset = initialize_dataset_experiments(
        args.dataset_name,
        device=args.device,
        dataset_type="arff",
        is_global=False,
    )
    data_train, data_valid, data_test = hmc_dataset.get_datasets()

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
        .to(args.device)
    )
    data_train.X = (
        torch.tensor(scaler.transform(imp_mean.transform(data_train.X)))
        .clone()
        .detach()
        .to(args.device)
    )
    data_test.X = (
        torch.as_tensor(scaler.transform(imp_mean.transform(data_test.X)))
        .clone()
        .detach()
        .to(args.device)
    )

    data_test.Y = torch.as_tensor(data_test.Y).clone().detach().to(args.device)
    data_valid.Y = torch.tensor(data_valid.Y).clone().detach().to(args.device)
    data_train.Y = torch.tensor(data_train.Y).clone().detach().to(args.device)

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
    args.input_dim = args.input_dims[args.data]
    args.max_depth = hmc_dataset.max_depth
    args.to_eval = hmc_dataset.to_eval
    if args.active_levels is None:
        args.active_levels = [i for i in range(args.max_depth)]
        logging.info(f"Active levels: {args.active_levels}")
    else:
        args.active_levels = [int(x) for x in args.active_levels]
        logging.info(f"Active levels: {args.active_levels}")

    criterions = [nn.BCELoss() for _ in hmc_dataset.levels_size]
    args.criterions = criterions

    if args.hpo == "true":
        logging.info("Hyperparameter optimization")
        args.n_trials = 20
        best_params = optimize_hyperparameters_per_level(args=args)

        logging.info(best_params)
    else:
        lr_values = [float(x) for x in args.lr_values]
        dropout_values = [float(x) for x in args.dropout_values]
        hidden_dims = [int(x) for x in args.hidden_dims]
        num_layers_values = [int(x) for x in args.num_layers_values]
        weight_decay_values = [float(x) for x in args.weight_decay_values]

        assert (
            len(lr_values)
            == len(dropout_values)
            == len(hidden_dims)
            == len(num_layers_values)
            == len(weight_decay_values)
            == args.max_depth
        ), "All hyperparameter lists must have the same length."

        params = {
            "levels_size": hmc_dataset.levels_size,
            "input_size": args.input_dims[args.data],
            "hidden_size": hidden_dims,
            "num_layers": num_layers_values,
            "dropout": dropout_values,
        }

        model = HMCLocalModel(**params)
        args.model = model
        logging.info(model)
        # Create the model
        # model = HMCLocalClassificationModel(levels_size=hmc_dataset.levels_size,
        #                                     input_size=args.input_dims[data],
        #                                     hidden_size=args.hidden_dim)
        run(args)
        test(args)
