import torch
import torch.nn as nn


class ExpandOutputClassification(nn.Module):
    def __init__(self, input_shape=512):
        super(ExpandOutputClassification, self).__init__()
        self.dense = nn.Linear(input_shape, input_shape)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dense(x)
        x = self.relu(x)
        return x


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
        self.dropouts = dropouts

        self.levels = nn.ModuleList()
        next_size = 0
        for size, dropout in zip(levels_size, dropouts):
            self.levels.append(BuildClassification(size, dropout, input_shape=sequence_size + next_size))
            next_size = size

    def forward(self, x):
        outputs = []
        current_input = x
        for i, level in enumerate(self.levels):
            if i != 0:
                current_input = torch.cat((current_output.detach(), x), dim=1)
            current_output = level(current_input)
            outputs.append(current_output)
        return outputs