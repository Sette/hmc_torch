import os
import torch
from torch.utils.data import Dataset
import ast
import pandas as pd
import numpy as np

BUFFER_SIZE = 10

# Função para converter a string em lista de np.array
def convert_to_list_of_arrays(val):
    # Converte string para uma lista de strings
    list_str = ast.literal_eval(val)
    # Converte cada item da lista para um array NumPy
    return [np.array(item) for item in list_str]


class HMCDataset(Dataset):
    def __init__(self, files, levels_size, testset=False):
        self.files = files
        self.levels_size = levels_size
        self.testset = testset
        self.data = self.load_data()
        
    def to_dataframe(self):
        records = []
        labels = {}
        for example in self.data:
            for level in range(1, len(self.levels_size) + 1):
                labels.update({f'level{level}': example[f'level{level}']})
            if self.testset:
                record = {'track_id': example['track_id'], 'features': example['features'], 'labels': labels}
            else:
                record = {'features': example['features'], 'labels': labels}
            records.append(record)
        return pd.DataFrame(records)

    def load_data(self):
        data = []
        for file in os.listdir(self.files):
            file_path = os.path.join(self.files, file)
            data.extend(torch.load(file_path, weights_only=False))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        features = torch.tensor(example['features'], dtype=torch.float32)
        labels = [torch.tensor(example[f'level{level}'], dtype=torch.float32)
            for level in range(1, len(self.levels_size) + 1)]
        if self.testset:
            track_id = torch.tensor(example['track_id'], dtype=torch.int64)
            return track_id, features, labels
        
        return features, labels


