import json
import os
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from hmc.model import ClassificationModel
from hmc.dataset import HMCDataset


class MaskedBCELoss(nn.Module):
    def __init__(self):
        super(MaskedBCELoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='none')  # Redução 'none' para manter a forma do tensor

    def forward(self, outputs, targets):
        losses = []
        for output, target in zip(outputs, targets):
            if len(target.shape) > 1:
                mask = target.sum(dim=1) > 0  # Dimensão 1 para targets 2D
            else:
                mask = target.sum() > 0  # Targets 1D ou outros casos

            if mask.any():
                loss = self.bce_loss(output, target)  # Calcula a perda sem redução
                masked_loss = loss[mask]  # Aplica a máscara
                losses.append(masked_loss.mean())  # Calcula a média da perda mascarada

        if len(losses) > 0:
            return torch.stack(losses).mean()  # Retorna um tensor e calcula a média
        else:
            return torch.tensor(0.0, requires_grad=True).to(outputs[0].device)  # Retorna uma perda zero se não houver perdas


def run():
    print("========================= PyTorch =========================")
    print("GPUs available: {}".format(torch.cuda.device_count()))
    print("===========================================================")

    parser = argparse.ArgumentParser(description='Train a classification model.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to data and metadata files.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the models.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs for training.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping.')

    args = parser.parse_args()

    metadata_path = os.path.join(args.input_path, 'metadata.json')
    labels_path = os.path.join(args.input_path, 'labels.json')

    with open(metadata_path, 'r') as f:
        metadata = json.loads(f.read())

    with open(labels_path, 'r') as f:
        labels = json.loads(f.read())

    params = {
        'levels_size': labels['levels_size'],
        'sequence_size': metadata['sequence_size'],
        'dropout': args.dropout
    }

    model = ClassificationModel(**params)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = MaskedBCELoss()  # Usando MaskedBCELoss

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    metadata['torch_path'] = os.path.join(args.input_path, 'torch')
    metadata['train_torch_path'] = os.path.join(args.torch_path, 'train')
    metadata['val_torch_path'] = os.path.join(args.torch_path, 'val')
    metadata['test_torch_path'] = os.path.join(args.torch_path, 'test')

    ds_train = HMCDataset(metadata['train_torch_path'], params['levels_size'])
    ds_validation = HMCDataset(metadata['val_torch_path'], params['levels_size'])

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ds_validation, batch_size=args.batch_size, shuffle=False)

    early_stopping_patience = args.patience
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        global_train_loss = 0.0
        local_train_losses = [0.0 for _ in range(metadata['max_depth'])]

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), [target.cuda() for target in targets]
            outputs = model(inputs)
            for index, (output, target) in enumerate(zip(outputs, targets)):
                loss = criterion(output, target)
                local_train_losses[index] = loss.detach()
                loss.backward()

            optimizer.step()
            global_train_loss += sum(loss.detach() for loss in local_train_losses) / len(local_train_losses)

        global_train_loss /= len(train_loader)
        local_train_losses = [loss/len(train_loader) for loss in local_train_losses]

        model.eval()
        global_val_loss = 0.0
        local_val_losses = [0.0 for _ in range(metadata['max_depth'])]
        with torch.no_grad():
            for inputs, targets in val_loader:
                if torch.cuda.is_available():
                    inputs, targets = inputs.cuda(), [target.cuda() for target in targets]
                outputs = model(inputs)
                for index, (output, target) in enumerate(zip(outputs, targets)):
                    loss = criterion(output, target)
                    local_val_losses[index] += loss.detach()
            global_val_loss += sum(loss.detach() for loss in local_val_losses) / len(local_val_losses)

        global_val_loss /= len(val_loader)
        local_val_losses = [loss/len(val_loader) for loss in local_val_losses]

        print(f'Epoch {epoch + 1}/{args.epochs}, Global Train Loss: {global_train_loss:.4f}, Global Validation Loss: {global_val_loss:.4f}')

        if global_val_loss < best_val_loss:
            best_val_loss = global_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.model_path, 'best_binary.pth'))
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break


