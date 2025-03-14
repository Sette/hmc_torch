import json
import os

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from hmc.model import HMCLocalClassificationModel
from hmc.dataset import HMCDataset
from hmc.model.losses import show_global_loss, show_local_losses
from hmc.utils.dir import create_job_id, create_dir
from hmc.model.arguments import get_parser


def train_local():
    print("========================= PyTorch =========================")
    print("GPUs available: {}".format(torch.cuda.device_count()))
    print("===========================================================")

    job_id = create_job_id()
    print(f"Job ID: {job_id}")

    parser = get_parser()
    args = parser.parse_args()

    dropouts = [float(rate) for rate in args.dropouts]
    thresholds = [float(threshold) for threshold in args.thresholds]

    metadata_path = os.path.join(args.input_path, 'metadata.json')
    labels_path = os.path.join(args.input_path, 'labels.json')

    with open(metadata_path, 'r') as f:
        metadata = json.loads(f.read())

    with open(labels_path, 'r') as f:
        labels = json.loads(f.read())

    params = {
        'levels_size': labels['levels_size'],
        'input_size': metadata['sequence_size'],
        'dropouts': dropouts,
        'thresholds': thresholds
    }

    assert len(args.dropouts) == metadata['max_depth']
    assert len(args.lrs) == metadata['max_depth']

    model = HMCLocalClassificationModel(**params)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

    criterions = [nn.BCEWithLogitsLoss(reduction='sum') for _ in labels['levels_size']]

    if torch.cuda.is_available():
        model = model.to('cuda')
        criterions = [criterion.to('cuda') for criterion in criterions]
        

    torch_path = os.path.join(args.input_path, 'torch')
    metadata['train_torch_path'] = os.path.join(torch_path, 'train')
    metadata['val_torch_path'] = os.path.join(torch_path, 'val')
    metadata['test_torch_path'] = os.path.join(torch_path, 'test')

    ds_train = HMCDataset(metadata['train_torch_path'], params['levels_size'])
    ds_validation = HMCDataset(metadata['val_torch_path'], params['levels_size'])

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ds_validation, batch_size=args.batch_size, shuffle=False)

    assert isinstance(args.output_path, str)
    model_path: str = os.path.join(args.output_path, 'jobs' ,job_id)
    create_dir(model_path)

    early_stopping_patience = args.patience
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs+1):
        model.train()
        local_train_losses = [0.0 for _ in range(metadata['max_depth'])]
        for inputs, targets in train_loader:
            if torch.cuda.is_available():
                inputs, targets = inputs.to('cuda'), [target.to('cuda') for target in targets]
            outputs = model(inputs)
            
            # Zerar os gradientes antes de cada batch
            optimizer.zero_grad()

            total_loss = 0.0
            for index, (output, target) in enumerate(zip(outputs, targets)):
                loss = criterions[index](output, target)
                total_loss += loss
                local_train_losses[index] += loss.item()

        # Backward pass (cálculo dos gradientes)
        total_loss.backward()
        
        optimizer.step()

        local_train_losses = [loss / len(train_loader) for loss in local_train_losses]
        global_train_loss = sum(local_train_losses) / metadata['max_depth']

        print(f'Epoch {epoch}/{args.epochs}')
        show_local_losses(local_train_losses, set='Train')
        show_global_loss(global_train_loss, set='Train')
            
        model.eval()
        local_val_losses = [0.0 for _ in range(metadata['max_depth'])]
        with torch.no_grad():
            for inputs, targets in val_loader:
                if torch.cuda.is_available():
                    inputs, targets = inputs.to('cuda'), [target.to('cuda') for target in targets]
                outputs = model(inputs)

                total_val_loss = 0.0
                for index, (output, target) in enumerate(zip(outputs, targets)):
                    loss = criterions[index](output, target)
                    total_val_loss += loss
                    local_val_losses[index] += loss.item() 

        local_val_losses = [loss / len(val_loader) for loss in local_val_losses]
        global_val_loss = sum(local_val_losses) / metadata['max_depth']

        print(f'Epoch {epoch}/{args.epochs}')
        show_local_losses(local_val_losses, set='Val')
        show_global_loss(global_val_loss, set='Val')

        current_val_loss = round(global_val_loss, 4)
        if current_val_loss <= best_val_loss - 2e-4:
            best_val_loss = current_val_loss
            print('new best model')
            torch.save(model.state_dict(), os.path.join(model_path, f'best_binary-{epoch}.pth'))
        else:
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered")
                return None
            
            patience_counter += 1
    return None


