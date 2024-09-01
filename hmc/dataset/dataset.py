import os
import torch
from torch.utils.data import Dataset

import numpy as np

BUFFER_SIZE = 10

class HMCDataset(Dataset):
    def __init__(self, files, levels_size):
        self.files = files
        self.levels_size = levels_size
        self.data = self.load_data()

    def load_data(self):
        # Carregar dados a partir dos arquivos e retornar uma lista de exemplos
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
        #track_id = example['track_id']
        labels = [torch.tensor(np.array(example[f'level{level}']), dtype=torch.float32)
                  for level in range(1, len(self.levels_size) + 1)]
        return  features, labels


