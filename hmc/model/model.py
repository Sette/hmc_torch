import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader
from hmc.dataset import HMCDataset
from hmc.model.metrics import custom_range

class ExpandOutputClassification(nn.Module):
    def __init__(self, input_shape=512):
        super(ExpandOutputClassification, self).__init__()
        self.dense = nn.Linear(input_shape, input_shape)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dense(x)
        x = self.relu(x)
        return x

class OutputNormalization(nn.Module):
    def forward(self, x):
        # Obtém os índices dos maiores valores ao longo da dimensão 1
        indices = torch.argmax(x, dim=1)
        # Converte esses índices para one-hot encoding
        return F.one_hot(indices, num_classes=x.size(1)).to(dtype=x.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape

class BuildClassification(nn.Module):
    def __init__(self, size, dropout, input_shape=1024):
        super(BuildClassification, self).__init__()
        self.fc1 = nn.Linear(input_shape, input_shape // 2)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(input_shape // 2, size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class ClassificationModel(nn.Module):
    def __init__(self, levels_size, sequence_size=1280, dropouts=[]):
        super(ClassificationModel, self).__init__()
        self.sequence_size = sequence_size
        self.levels_size = levels_size
        self.thesholds = custom_range(len(levels_size))
        self.dropouts = dropouts
        self.levels = nn.ModuleList()
        self.output_normalization = OutputNormalization()
        next_size = 0
        for size, dropout in zip(levels_size, dropouts):
            self.levels.append(BuildClassification(size, dropout, input_shape=sequence_size + next_size))
            next_size = size

    def forward(self, x):
        outputs = []
        current_input = x
        current_output = current_input
        for i, level in enumerate(self.levels):
            if i != 0:
                current_input = torch.cat((current_output.detach(), x), dim=1)
            current_output = level(current_input)
            outputs.append(current_output)
            current_output = self.output_normalization(current_output)
        assert isinstance(outputs, object)
        return outputs

    def predict(self, testset_path, batch_size=64):
        self.eval()  # Coloca o modelo em modo de avaliação
        ds_test = HMCDataset(testset_path, self.levels_size)
        test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
        predictions = []

        with torch.no_grad():  # Desativa o cálculo de gradientes
            for inputs, _ in test_loader:
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                binary_outputs = [(output >= 0.6).cpu().detach().numpy().astype(int) for output in self(inputs)]
                predictions.append(binary_outputs)

        predictions = [np.vstack(level_targets) for level_targets in zip(*predictions)]
        return predictions