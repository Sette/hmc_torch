import json
import os
import torch
from torch.utils.data import DataLoader
from hmc.model import ClassificationModel
from hmc.dataset import HMCDataset
from hmc.model.losses import MaskedBCELoss, WeightedMaskedBCELoss, hierarchical_loss, show_global_loss, show_local_losses
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

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

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

    # Definir pesos para cada nível hierárquico
    level_weights = [1.0, 0.8, 0.6, 0.4, 0.2]  # Ajuste conforme necessário

    # Modificar o critério para incluir os pesos
    criterion = WeightedMaskedBCELoss(level_weights)

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    # ... (resto do código permanece o mesmo até o loop de treinamento)

    for epoch in range(1, args.epochs + 1):
        model.train()
        global_train_loss = 0.0
        local_train_losses = [0.0 for _ in range(metadata['max_depth'])]
        hierarchical_train_loss = 0.0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), [target.cuda() for target in targets]
            outputs = model(inputs)

            # Calcular loss hierárquica
            h_loss = hierarchical_loss(outputs, targets, level_weights)
            hierarchical_train_loss += h_loss.item()

            # Calcular losses locais e global
            for index, (output, target) in enumerate(zip(outputs, targets)):
                loss = criterion(output, target, index)
                local_train_losses[index] += loss.item()
                global_train_loss += loss.item() * level_weights[index]

            # Backpropagation usando a loss hierárquica
            h_loss.backward()
            optimizer.step()

        # Normalizar as losses
        num_batches = len(train_loader)
        hierarchical_train_loss /= num_batches
        global_train_loss /= num_batches
        local_train_losses = [loss / num_batches for loss in local_train_losses]

        print(f'Epoch {epoch}/{args.epochs}')
        show_local_losses(local_train_losses, set='train')
        show_global_loss(global_train_loss, set='train')
        print(f'Hierarchical Train Loss: {hierarchical_train_loss:.4f}')

        # Validação
        model.eval()
        global_val_loss = 0.0
        local_val_losses = [0.0 for _ in range(metadata['max_depth'])]
        hierarchical_val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                if torch.cuda.is_available():
                    inputs, targets = inputs.cuda(), [target.cuda() for target in targets]
                outputs = model(inputs)

                # Calcular loss hierárquica
                h_loss = hierarchical_loss(outputs, targets, level_weights)
                hierarchical_val_loss += h_loss.item()

                # Calcular losses locais e global
                for index, (output, target) in enumerate(zip(outputs, targets)):
                    loss = criterion(output, target, index)
                    local_val_losses[index] += loss.item()
                    global_val_loss += loss.item() * level_weights[index]

        # Normalizar as losses
        num_val_batches = len(val_loader)
        hierarchical_val_loss /= num_val_batches
        global_val_loss /= num_val_batches
        local_val_losses = [loss / num_val_batches for loss in local_val_losses]

        print(f'Epoch {epoch}/{args.epochs}')
        show_local_losses(local_val_losses, set='val')
        show_global_loss(global_val_loss, set='val')
        print(f'Hierarchical Val Loss: {hierarchical_val_loss:.4f}')

        # Early stopping baseado na loss hierárquica
        if hierarchical_val_loss < best_val_loss:
            best_val_loss = hierarchical_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(model_path, 'best_binary.pth'))
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break



