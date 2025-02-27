import os

import torch, random, argparse
import torch.utils.data
import numpy as np

from hmc.model.train_global import train_global
from hmc.model.arguments import get_parser

def train():
    # Training settings
    parser = get_parser()
    args = parser.parse_args()

    args.hyperparams = {'batch_size': args.batch_size, 'num_layers': args.num_layers, 'dropout': args.dropout,
                   'non_lin': args.non_lin, 'hidden_dim': args.hidden_dim, 'lr': args.lr,
                   'weight_decay': args.weight_decay}
    ## Insert her a logic to use all datasets with arguments

    if 'all' in args.datasets:
        datasets = ['cellcycle_GO', 'derisi_GO', 'eisen_GO', 'expr_GO', 'gasch1_GO',
                    'gasch2_GO', 'seq_GO', 'spo_GO', 'cellcycle_FUN', 'derisi_FUN',
                    'eisen_FUN', 'expr_FUN', 'gasch1_FUN', 'gasch2_FUN', 'seq_FUN', 'spo_FUN']
    else:
        if len(args.datasets) > 1:
            datasets = [str(dataset) for dataset in args.datasets]
        else:
            datasets = [args.datasets]
            assert ('_' in args.datasets)
            assert ('FUN' in args.datasets or 'GO' in args.datasets or 'others' in args.datasets)

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
