import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from hmc.dataset.manager.dataset_manager import initialize_dataset_experiments
from hmc.model.local_classifier.baseline.model import HMCLocalModel
from hmc.train.local_classifier.baseline.hpo.hpo_local import (
    optimize_hyperparameters_per_level,
)

from hmc.train.utils import (
    show_global_loss,
    show_local_losses,
    show_local_precision,
    create_job_id_name,
    save_dict_to_json,
)
from hmc.utils.dir import create_dir


def train_step(args):
    """
    Executes the training loop for a hierarchical multi-class (HMC) local classifier model.
    This function performs the following steps:
    - Moves the model and loss criterions to the specified device.
    - Initializes early stopping parameters and tracking variables for each level of the hierarchy.
    - Sets up optimizers for each model level with individual learning rates and weight decays.
    - Iterates over the specified number of epochs, performing:
        - Training over batches: forward pass, loss computation for active levels, and gradient accumulation.
        - Backward pass and optimizer step for each level.
        - Logging of training losses.
        - Periodic evaluation on the validation set, including loss and precision reporting.
        - Early stopping if all levels have triggered it.
    Args:
        args: An object containing all necessary training parameters and objects, including:
            - model: The hierarchical model with per-level submodules.
            - criterions: List of loss functions for each level.
            - device: Device to run computations on.
            - hmc_dataset: Dataset object with max_depth attribute.
            - active_levels: List of currently active levels for training.
            - max_depth: Maximum depth of the hierarchy.
            - lr_values: List of learning rates for each level.
            - weight_decay_values: List of weight decay values for each level.
            - epochs: Number of training epochs.
            - train_loader: DataLoader for training data.
            - epochs_to_evaluate: Frequency of validation evaluation.
            - Additional attributes used for logging and early stopping.
    """

    args.model = args.model.to(args.device)
    args.criterions = [criterion.to(args.device) for criterion in args.criterions]

    args.early_stopping_patience = 3
    args.patience_counters = [0] * args.hmc_dataset.max_depth
    # args.level_active = [True] * args.hmc_dataset.max_depth
    args.level_active = [level in args.active_levels for level in range(args.max_depth)]
    logging.info("Active levels: %s", args.active_levels)
    logging.info("Level active: %s", args.level_active)

    args.best_val_loss = [float("inf")] * args.max_depth
    args.best_model = [None] * args.max_depth
    logging.info("Best val loss created %s", args.best_val_loss)

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
        logging.info("Active levels: %s", args.active_levels)

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
        for idx, total_loss in enumerate(local_train_losses):
            if total_loss > 0 and args.level_active[idx]:
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

        logging.info("Epoch %d/%d", epoch, args.epochs)
        show_local_losses(local_train_losses, set="Train")
        show_global_loss(global_train_loss, set="Train")

        if epoch % args.epochs_to_evaluate == 0:
            local_val_losses, local_val_precision = val_step(args)
            show_local_losses(local_val_losses, set="Val")
            show_local_precision(local_val_precision, set="Val")

            if not any(args.level_active):
                logging.info("All levels have triggered early stopping.")
                for i in args.active_levels:
                    args.model.levels[i].load_state_dict(args.best_model[i])
                break


def val_step(args):
    """
    Performs a validation step for a hierarchical multi-level classifier model.
    Args:
        args: An object containing all necessary arguments and attributes, including:
            - model: The model to evaluate, with attribute `levels` for each depth.
            - val_loader: DataLoader for the validation dataset.
            - criterions: List of loss functions, one per level.
            - device: Device to run computations on.
            - max_depth: Maximum number of levels in the hierarchy.
            - active_levels: List of indices for currently active levels.
            - best_val_loss: List of best validation losses per level.
            - best_model: List to store the best model state_dict per level.
            - patience_counters: List of patience counters for early stopping per level.
            - early_stopping_patience: Number of epochs to wait before early stopping.
            - level_active: List indicating if a level is still active.
    Returns:
        tuple:
            - local_val_losses (list of float): Average validation loss per level.
            - local_val_precision (list of float): Average precision score per level.
    Side Effects:
        - Updates `args.best_val_loss`, `args.best_model`, and `args.patience_counters` for improved levels.
        - Freezes parameters of levels that triggered early stopping by setting `requires_grad` to False.
        - Logs progress and early stopping events.
    """

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
    logging.info("Levels to evaluate: %s", args.active_levels)
    for i in args.active_levels:
        if args.level_active[i]:
            if args.best_model[i] is None:
                args.best_model[i] = args.model.levels[i].state_dict()
                logging.info("Level %d: initialized best model", i)
            if round(local_val_losses[i].item(), 4) < args.best_val_loss[i]:
                args.best_val_loss[i] = round(local_val_losses[i].item(), 4)
                args.best_model[i] = args.model.levels[i].state_dict()
                args.patience_counters[i] = 0
                logging.info("Level %d: improved (loss=%.4f)", i, local_val_losses[i])
            else:
                args.patience_counters[i] += 1
                logging.info(
                    "Level %d: no improvement (patience %d/%d)",
                    i,
                    args.patience_counters[i],
                    args.early_stopping_patience,
                )
                if args.patience_counters[i] >= args.early_stopping_patience:
                    args.level_active[i] = False
                    # args.active_levels.remove(i)
                    logging.info(
                        "🚫 Early stopping triggered for level %d — freezing its parameters",
                        i,
                    )
                    # ❄️ Congelar os parâmetros desse nível
                    for param in args.model.levels[i].parameters():
                        param.requires_grad = False
    return local_val_losses, local_val_precision


def test_step(args):
    """
    Evaluates the model on the test dataset for each active level and saves the results.
    Args:
        args: An object containing the following attributes:
            - model: The trained model to evaluate.
            - test_loader: DataLoader providing test data batches (inputs, targets, global_targets).
            - device: The device (CPU or CUDA) to run computations on.
            - active_levels: Iterable of indices indicating which levels to evaluate.
            - dataset_name: Name of the dataset (used for saving results).
            - hmc_dataset: Dataset object containing hierarchical information (optional, for global evaluation).
    Returns:
        None. The function saves the evaluation results as a JSON file in the 'results/train' directory.
    Side Effects:
        - Logs evaluation progress and results.
        - Saves local test scores (precision, recall, f-score, support) for each active level to a JSON file.
    """

    args.model.eval()
    local_inputs = {level: [] for _, level in enumerate(args.active_levels)}
    local_outputs = {level: [] for _, level in enumerate(args.active_levels)}

    threshold = 0.3

    Y_true_global = []
    with torch.no_grad():
        for inputs, targets, global_targets in args.test_loader:
            inputs = inputs.to(args.device)
            targets = [target.to(args.device).float() for target in targets]
            global_targets = global_targets.to("cpu")
            outputs = args.model(inputs.float())

            for index, (output, target) in enumerate(zip(outputs, targets)):
                if index in args.active_levels:
                    output = output.to("cpu")
                    target = target.to("cpu")
                    local_inputs[index].append(target)
                    local_outputs[index].append(output)
            Y_true_global.append(global_targets)
        # Concat all outputs and targets by level
    local_inputs = {
        level: torch.cat(local_input, dim=0)
        for level, local_input in local_inputs.items()
    }
    local_outputs = {
        key: torch.cat(outputs, dim=0) for key, outputs in local_outputs.items()
    }
    # Get local scores
    local_test_score = {level: None for _, level in enumerate(args.active_levels)}

    logging.info("Evaluating %d active levels...", len(args.active_levels))
    for idx in args.active_levels:
        y_pred_binary = local_outputs[idx].data > threshold

        # y_pred_binary = (local_outputs[idx] > threshold).astype(int)

        score = precision_recall_fscore_support(
            local_inputs[idx], y_pred_binary, average="micro"
        )

        local_test_score[idx] = score

    logging.info("Local test score: %s", str(local_test_score))

    job_id = create_job_id_name(prefix="test")

    create_dir("results/train")

    save_dict_to_json(
        local_test_score,
        f"results/train/{args.dataset_name}-{job_id}.json",
    )

    # Save the trained model
    torch.save(args.model.state_dict(), f"results/train/{args.dataset_name}-{job_id}-state_dict.pt")
    # args.model.save(f"results/train/{args.dataset_name}-{job_id}.pt")

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


def train_local(args):
    """
    Trains a local hierarchical multi-label classifier using the specified arguments.
    This function sets up the experiment environment, loads and preprocesses the dataset,
    creates data loaders for training, validation, and testing, initializes loss functions,
    and either performs hyperparameter optimization or trains the model with provided hyperparameters.
    Args:
        args: An argparse.Namespace or similar object containing the following attributes:
            - dataset_name (str): Name of the dataset in the format "data_ontology".
            - device (str): Device to use ("cpu" or "cuda").
            - batch_size (int): Batch size for data loaders.
            - input_dims (dict): Dictionary mapping dataset names to input dimensions.
            - hpo (str): Whether to perform hyperparameter optimization ("true" or "false").
            - lr_values (list): List of learning rates per level.
            - dropout_values (list): List of dropout rates per level.
            - hidden_dims (list): List of hidden layer sizes per level.
            - num_layers_values (list): List of number of layers per level.
            - weight_decay_values (list): List of weight decay values per level.
            - active_levels (list or None): List of active levels to train, or None for all.
            - Other attributes required by downstream functions.
    Side Effects:
        - Updates the `args` object with data loaders, dataset information, loss functions, and model.
        - Logs experiment information and progress.
    Raises:
        AssertionError: If the lengths of hyperparameter lists do not match the number of levels.
    """

    logging.info(".......................................")
    logging.info("Experiment with %s dataset", args.dataset_name)
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
    else:
        args.active_levels = [int(x) for x in args.active_levels]
    logging.info("Active levels: %s", args.active_levels)

    criterions = [nn.BCELoss() for _ in hmc_dataset.levels_size]
    args.criterions = criterions

    if args.hpo == "true":
        logging.info("Hyperparameter optimization")
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
            "active_levels": args.active_levels,
        }

        model = HMCLocalModel(**params)
        args.model = model
        logging.info(model)
        # Create the model
        # model = HMCLocalClassificationModel(levels_size=hmc_dataset.levels_size,
        #                                     input_size=args.input_dims[data],
        #                                     hidden_size=args.hidden_dim)
        train_step(args)
        test_step(args)
