import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Train a Hierarchical Multi-label Classification model.')
    # Dataset parameters
    parser.add_argument('--datasets',
                        type=str, nargs='+',
                        default=['seq_GO', 'derisi_GO', 'gasch1_GO', 'cellcycle_FUN'],
                        help='List with dataset names.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to data and metadata files.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the models.')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate by level.')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs for training.')
    parser.add_argument('--dropout', type=float,  default=0.7, help='Dropout rate.')

    # Model architecture parameters
    parser.add_argument('--hidden_dim', type=int, default=50, help='Dimension of hidden layers.')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the model.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 regularization).')
    parser.add_argument('--non_lin', type=str, default='relu', choices=['relu', 'tanh', 'sigmoid'],
                        help='Non-linearity function.')

    # Hardware and execution parameters
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (e.g., "cpu" or "cuda").')
    parser.add_argument('--num_epochs', type=int, default=2000, help='Total number of training epochs.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')

    # Method parameters
    parser.add_argument('--method', type=str, default='global', choices=['global', 'local'],
                        help='Method type to use.')
    return parser

