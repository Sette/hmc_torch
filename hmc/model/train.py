import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import numpy as np
from hmc.model import MusicModel
from hmc.dataset import MusicDataset
import sys
import types

import json

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
    
    return running_loss / len(dataloader)


print("========================= Torch =========================")
print("GPUs availables: {}".format(torch.cuda.is_available()))
print("==============================================================")


def run(args: object):
    print(args)

    with open(args.metadata_path, 'r') as f:
        metadata = json.loads(f.read())
        print(metadata)

    with open(args.labels_path, 'r') as f:
        labels = json.loads(f.read())

    levels_size = {'level1': labels['label_1_count'] ,
                   'level2': labels['label_2_count'] ,
                   'level3': labels['label_3_count'] ,
                   'level4': labels['label_4_count'] }

    params: dict = {
        'levels_size': levels_size,
        'sequence_size': metadata['sequence_size'],
        'dropout': args.dropout
    }

    print(params)
    model = MusicModel(levels_size)
    # Exemplo de uso
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.BCELoss()  # ou outra loss apropriada para seu caso
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create a dummy dataset module with dataset_torch submodule
    dataset = types.ModuleType('dataset')
    dataset_torch = types.ModuleType('dataset_torch')
    dataset.dataset_torch = dataset_torch
    sys.modules['dataset'] = dataset
    sys.modules['dataset.dataset_torch'] = dataset_torch


    dataset_torch.MusicDataset = MusicDataset
    sys.modules['dataset_torch'] = dataset_torch



    # Carregar o dataset salvo
    train_dataset = torch.load(args.train_path)
    val_dataset = torch.load(args.val_path)
    
    # Criação do DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    
    # Treinamento e avaliação
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, criterion, optimizer, device)
        val_loss = evaluate(model, val_dataloader, criterion, device)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Validation Loss: {val_loss}')
