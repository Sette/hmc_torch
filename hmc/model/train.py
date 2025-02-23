import os

import torch, random, argparse
import torch.utils.data
import numpy as np

from hmc.model.train_global import train_global


def train():
    # Training settings
    parser = argparse.ArgumentParser(description='Train neural network')

    # Required  parameters
    parser.add_argument('--dataset', type=str, required=True,
                        nargs='+', default=['seq_GO', 'derisi_GO', '0.6', '0.7'],
                        help='List with dataset names to train')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='dataset path')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='input batch size for training')
    parser.add_argument('--lr', type=float, required=True,
                        help='learning rate')
    parser.add_argument('--dropout', type=float, required=True,
                        help='dropout probability')
    parser.add_argument('--hidden_dim', type=int, required=True,
                        help='size of the hidden layers')
    parser.add_argument('--num_layers', type=int, required=True,
                        help='number of hidden layers')
    parser.add_argument('--weight_decay', type=float, required=True,
                        help='weight decay')
    parser.add_argument('--non_lin', type=str, required=True,
                        help='non linearity function to be used in the hidden layers')
    parser.add_argument('--output_path', type=str, required=True,
                        help='output path')
    parser.add_argument('--device', type=int, default=0,
                        help='device (default:0)')
    parser.add_argument('--num_epochs', type=int, default=2000,
                        help='Max number of epochs to train (default:2000)')
    parser.add_argument('--method', type=str,  default="global", required=True,
                        help='train method (local or global)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default:0)')

    args = parser.parse_args()
    args.hyperparams = {'batch_size': args.batch_size, 'num_layers': args.num_layers, 'dropout': args.dropout,
                   'non_lin': args.non_lin, 'hidden_dim': args.hidden_dim, 'lr': args.lr,
                   'weight_decay': args.weight_decay}
    ## Insert her a logic to use all datasets with arguments

    if 'all' in args.dataset:
        datasets = ['cellcycle_GO', 'derisi_GO', 'eisen_GO', 'expr_GO', 'gasch1_GO',
                    'gasch2_GO', 'seq_GO', 'spo_GO', 'cellcycle_FUN', 'derisi_FUN',
                    'eisen_FUN', 'expr_FUN', 'gasch1_FUN', 'gasch2_FUN', 'seq_FUN', 'spo_FUN']
    else:
        if len(args.dataset) > 1:
            datasets = [str(dataset) for dataset in args.dataset]
        else:
            datasets = [args.dataset]
            assert ('_' in args.dataset)
            assert ('FUN' in args.dataset or 'GO' in args.dataset or 'others' in args.dataset)

    # Dictionaries with number of features and number of labels for each dataset
    args.input_dims = {'diatoms': 371, 'enron': 1001, 'imclef07a': 80, 'imclef07d': 80, 'cellcycle': 77, 'church': 27,
                  'derisi': 63, 'eisen': 79, 'expr': 560, 'gasch1': 173, 'gasch2': 52, 'hom': 47034, 'seq': 529,
                  'spo': 86}
    args.output_dims_FUN = {'cellcycle': 499, 'church': 499, 'derisi': 499, 'eisen': 461, 'expr': 499, 'gasch1': 499,
                       'gasch2': 499, 'hom': 499, 'seq': 499, 'spo': 499}
    args.output_dims_GO = {'cellcycle': 4122, 'church': 4122, 'derisi': 4116, 'eisen': 3570, 'expr': 4128, 'gasch1': 4122,
                      'gasch2': 4128, 'hom': 4128, 'seq': 4130, 'spo': 4116}
    args.output_dims = {'FUN': args.output_dims_FUN, 'GO': args.output_dims_GO}

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Verifica quantas GPUs estão disponíveis
    num_gpus = torch.cuda.device_count()
    print(f"Total de GPUs disponíveis: {num_gpus}")

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for dataset_name in datasets:
        if args.method == "global":
            train_global(dataset_name, args)
