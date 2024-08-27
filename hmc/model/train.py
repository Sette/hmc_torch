import json
import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from hmc.model import ClassificationModel
from hmc.dataset import HMCDataset
from hmc.model.losses import MaskedBCELoss, show_global_loss, show_local_losses
from hmc.utils.dir import create_job_id, create_dir
from hmc.model.arguments import get_parser

def run():
    print("========================= PyTorch =========================")
    print("GPUs available: {}".format(torch.cuda.device_count()))
    print("===========================================================")

    job_id = create_job_id()
    print(f"Job ID: {job_id}")

    parser = get_parser()
    args = parser.parse_args()

    dropouts = [float(rate) for rate in args.dropout]

    metadata_path = os.path.join(args.input_path, 'metadata.json')
    labels_path = os.path.join(args.input_path, 'labels.json')

    with open(metadata_path, 'r') as f:
        metadata = json.loads(f.read())

    with open(labels_path, 'r') as f:
        labels = json.loads(f.read())

    params = {
        'levels_size': labels['levels_size'],
        'sequence_size': metadata['sequence_size'],
        'dropouts': dropouts
    }

    assert len(args.dropout) == metadata['max_depth']

    model = ClassificationModel(**params)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = MaskedBCELoss()  # Usando MaskedBCELoss

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    torch_path = os.path.join(args.input_path, 'torch')
    metadata['train_torch_path'] = os.path.join(torch_path, 'train')
    metadata['val_torch_path'] = os.path.join(torch_path, 'val')
    metadata['test_torch_path'] = os.path.join(torch_path, 'test')

    ds_train = HMCDataset(metadata['train_torch_path'], params['levels_size'])
    ds_validation = HMCDataset(metadata['val_torch_path'], params['levels_size'])

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ds_validation, batch_size=args.batch_size, shuffle=False)

    model_path = os.path.join(args.output_path, job_id)
    create_dir(model_path)

    early_stopping_patience = args.patience
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs+1):
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
                loss.backward()
                optimizer.step()
                local_train_losses[index] += loss.item()

            global_train_loss = sum(local_train_losses)

        global_train_loss /= len(train_loader)
        local_train_losses = [loss / len(train_loader) for loss in local_train_losses]

        print(f'Epoch {epoch}/{args.epochs}')
        show_local_losses(local_train_losses, set='train')
        show_global_loss(global_train_loss, set='train')

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
                    local_val_losses[index] += loss.item()

                global_val_loss += sum(local_val_losses)

        global_val_loss /= len(val_loader)
        local_val_losses = [loss / len(val_loader) for loss in local_val_losses]

        print(f'Epoch {epoch}/{args.epochs}')
        show_local_losses(local_val_losses, set='val')
        show_global_loss(global_val_loss, set='val')

        if global_val_loss < best_val_loss:
            best_val_loss = global_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(model_path, 'best_binary.pth'))
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break
