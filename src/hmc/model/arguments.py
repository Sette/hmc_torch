import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        description="Train a Hierarchical Multi-label Classification model."
    )
    # Dataset parameters
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["seq_GO", "derisi_GO", "gasch1_GO", "cellcycle_FUN"],
        help="List with dataset names.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to data and metadata files.",
    )

    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save the models."
    )

    # Training parameters
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training."
    )

    parser.add_argument(
        "--epochs", type=int, default=15, help="Number of epochs for training."
    )

    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["csv", "torch", "arff"],
        required=False,
        default="arff",
        metavar="DATASET_TYPE",
        help="Type of dataset to load.",
    )

    parser.add_argument(
        "--non_lin",
        type=str,
        default="relu",
        choices=["relu", "tanh", "sigmoid"],
        help="Non-linearity function.",
    )

    # Hardware and execution parameters
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Device to use (e.g., "cpu" or "cuda").',
    )
    parser.add_argument(
        "--num_epochs", type=int, default=2000, help="Total number of training epochs."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility."
    )

    # Method parameters
    parser.add_argument(
        "--method",
        type=str,
        default="global",
        choices=["global", "local", "globalLM", "global_baseline"],
        metavar="METHOD",
        help="Method type to use.",
    )

    # Hyperparameter Optimization (HPO) parameters
    parser.add_argument(
        "--hpo",
        type=str,
        default="false",
        choices=["true", "false"],
        metavar="HPO",
        help="Enable or disable Hyperparameter Optimization (HPO). \
            Use 'true' to enable and 'false' to disable.",
    )

    # HPO result parameters (used when HPO is disabled)
    parser.add_argument(
        "--lr_values",
        type=float,
        nargs="+",
        required=False,
        help="List of values for the learning rate (used when HPO is disabled).",
    )
    parser.add_argument(
        "--dropout_values",
        type=float,
        nargs="+",
        required=False,
        metavar="DROPOUT",
        help="List of values for dropout (used when HPO is disabled).",
    )
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        required=False,
        metavar="HIDDEN_DIMS",
        help="List of values for the number of hidden neurons (used when HPO is disabled).",
    )
    parser.add_argument(
        "--num_layers_values",
        type=int,
        nargs="+",
        required=False,
        metavar="NUM_LAYERS",
        help="List of values for the number of layers (used when HPO is disabled).",
    )
    parser.add_argument(
        "--weight_decay_values",
        type=float,
        nargs="+",
        required=False,
        metavar="WEIGHT_DECAY",
        help="List of values for weight decay (used when HPO is disabled).",
    )

    return parser
