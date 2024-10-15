import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os

from torch.utils.data import DataLoader
from hmc.dataset import HMCDataset
from hmc.model.metrics import custom_thresholds, custom_dropouts, custom_lrs

def transform_predictions(predictions):
    transformed = []
    # Loop through each index to form examples with the first element from each level
    for i in range(len(predictions[0])):  # Iterate over the number of examples
        example = []
        for level in predictions:  # Iterate over the levels
            example.append(level[i])  # Get the first element from each level at index i
        transformed.append(example)
    
    return transformed


class ExpandOutputClassification(nn.Module):
    def __init__(self, input_shape=512, output_shape=512):
        super().__init__()
        self.dense = nn.Linear(input_shape, output_shape)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dense(x)
        x = self.relu(x)
        return x

class OneHotOutputNormalization(nn.Module):
    def __init__(self, num_classes, threshold=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.threshold = threshold

    def forward(self, x):
        # Se o valor for maior que o limiar, a classe é considerada ativa (multi-label)
        binary_output = (x >= self.threshold).int()
        return binary_output.to(dtype=x.dtype, device=x.device)


class BuildClassification(nn.Module):
    def __init__(self, size, dropout, input_shape=1024):
        super(BuildClassification, self).__init__()
        self.fc1 = nn.Linear(input_shape, input_shape // 2)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(input_shape // 2, size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


class ClassificationModel(nn.Module):
    def __init__(self, levels_size, sequence_size=1280, dropouts=None, thresholds=None, lrs=None):
        super().__init__()
        self.sequence_size = sequence_size
        self.levels_size = levels_size
        if not thresholds:
            self.thresholds = custom_thresholds(len(levels_size))
        else:
            self.thresholds = thresholds
        if not dropouts:
            self.dropouts = custom_dropouts(len(levels_size))
        else:
            self.dropouts = dropouts
        if not lrs:
            self.lrs = custom_lrs(len(levels_size))
        else:
            self.lrs = lrs
        self.levels = nn.ModuleList()
        for size, dropout in zip(levels_size, dropouts):
            self.levels.append(BuildClassification(size, dropout, input_shape=sequence_size))

    def forward(self, x):
        outputs = []
        current_input = x
        for level in self.levels:
            local_output = level(current_input)
            outputs.append(local_output)
        return outputs
    
    def predict(self, base_path, batch_size=64):
        torch_path = os.path.join(base_path, 'torch')
        test_torch_path = os.path.join(torch_path, 'test')
        #test_csv_path = os.path.join(base_path, 'test.csv')
        self.eval()  
        ds_test = HMCDataset(test_torch_path, self.levels_size, testset=True)
        #df_test = pd.read_csv(test_csv_path)
        test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
        predictions = []
        with torch.no_grad():
            for track_id, inputs, _ in test_loader:
                # Para armazenar as saídas binárias de cada batch
                batch_predictions = []
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                # Recebe saídas para todos os níveis
                outputs_per_level = self(inputs)
                #print(outputs_per_level)
                levels_pred = {}
                for level, pred in enumerate(outputs_per_level, start=1):
                    level_name = f'level{level}'
                    levels_pred[level_name] = pred
                return track_id, levels_pred
                # Itera sobre as saídas de cada nível e aplica o threshold correspondente
                #for level_output, _ in zip(outputs_per_level, self.thresholds):
                    # Aplica o threshold para converter em saída binária (0 ou 1)
                    #binary_output = (level_output >= threshold).float()  #  threshold
                    #batch_predictions.append(binary_output.cpu().detach().numpy())  # Converte para NumPy e armazena
                #    batch_predictions.append(level_output.cpu().detach().numpy())
                    # Armazena as previsões do batch atual para todos os níveis
            #predictions.append(batch_predictions)
            #output_list = [level_targets for level_targets in zip(*predictions)]
            #output_list = transform_predictions(output_list)

        #df_test['predictions'] = output_list
        return predictions