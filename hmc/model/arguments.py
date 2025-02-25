import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Train a classification model.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to data and metadata files.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the models.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs for training.')
    parser.add_argument('--dropouts', type=str, nargs='+', default=['0.3', '0.5', '0.6', '0.7'], help='Dropout rate by level.')
    parser.add_argument('--thresholds', type=str, nargs='+', default=['0.3', '0.5', '0.6', '0.7'], help='Threshold rate by level.')
    parser.add_argument('--optimizers', type=str, nargs='+', default=['0.001', '0.001', '0.001', '0.001'], help='List with the optimizers fution names by level.')
    parser.add_argument('--lrs', type=str, nargs='+', default=['0.001', '0.001', '0.001', '0.001'], help='Learning rate by level.')
    parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping.')
    return parser