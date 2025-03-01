from hmc.model import ConstrainedFFNNModel, get_constr_out
from hmc.utils.dir import create_dir
from hmc.dataset import initialize_dataset
from hmc.env import SRC_LOG_LEVELS
import os
import logging
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from tqdm import tqdm
from sklearn.metrics import average_precision_score

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["TRAIN"])


def train_global(dataset_name, args):
    print(".......................................")
    print("Experiment with {} dataset ".format(dataset_name))
    # Load train, val and test set
    data = dataset_name.split('_')[0]
    ontology = dataset_name.split('_')[1]
    hmc_dataset = initialize_dataset(dataset_name, args.dataset_path, output_path=args.output_path, is_global=True)
    to_eval = torch.tensor(hmc_dataset.to_eval, dtype=torch.bool)
    train, val, test = hmc_dataset.get_torch_dataset()

    R = hmc_dataset.compute_matrix_R()
    device = torch.device(args.device)

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False)

    num_epochs = args.num_epochs
    if 'GO' in dataset_name:
        num_to_skip = 4
    else:
        num_to_skip = 1

    # Create the model
    model = ConstrainedFFNNModel(args.input_dims[data], args.hidden_dim,
                                 args.output_dims[ontology][data] + num_to_skip,
                                 args.hyperparams, R)
    model = model.to(device)
    to_eval = to_eval.to(device)
    R = R.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss()

    # Set patience
    patience, max_patience = 20, 20
    max_score = 0.0

    for epoch in tqdm(range(num_epochs)):
        model.train()

        for i, (x, labels) in enumerate(train_loader):
            x = x.to(device)
            labels = labels.to(device)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            output = model(x.float())

            constr_output = get_constr_out(output, R)
            train_output = labels * output.double()
            train_output = get_constr_out(train_output, R)
            train_output = (1 - labels) * constr_output.double() + labels * train_output

            #loss = criterion(train_output.float(), labels)
            loss = criterion(train_output[:, to_eval].float(), labels[:, to_eval])
            # Getting gradients w.r.t. parameters
            loss.backward()
            # Updating parameters
            optimizer.step()

        model.eval()

        for i, (x, y) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)

            constrained_output = model(x.float())
            predicted = constrained_output.data > 0.5
            # Move output and label back to cpu to be processed by sklearn
            predicted = predicted.to('cpu')
            cpu_constrained_output = constrained_output.to('cpu')
            y = y.to('cpu')
            to_eval = to_eval.to('cpu')

            if i == 0:
                predicted_test = predicted
                constr_test = cpu_constrained_output
                y_val = y
            else:
                predicted_test = torch.cat((predicted_test, predicted), dim=0)
                constr_test = torch.cat((constr_test, cpu_constrained_output), dim=0)
                y_val = torch.cat((y_val, y), dim=0)

        score = average_precision_score(y_val[:, to_eval], constr_test.data[:, to_eval], average='micro')
        if score >= max_score:
            patience = max_patience
            max_score = score
        else:
            patience = patience - 1

        create_dir(f'results/{str(args.job_id+dataset_name)}')
        f = open(f'results/{str(args.job_id+dataset_name)}/'+ "result_val"+ '.csv', 'a')
        f.write(str(score) + ',' + str(epoch) + ',' + str(args.seed)+ '\n')
        f.close()
        if patience == 0:
            break