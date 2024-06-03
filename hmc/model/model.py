import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class OutputNormalization(nn.Module):
    def forward(self, x):
        return F.one_hot(torch.argmax(x, dim=1), num_classes=x.shape[1]).float()
    

class ClassificationBlock(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super(ClassificationBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size // 2)
        self.fc3 = nn.Linear(input_size // 2, input_size // 4)
        self.fc4 = nn.Linear(input_size // 4, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc4(x))
        return x

class MusicModel(nn.Module):
    def __init__(self, levels_size, sequence_size=1280, dropout=0.1):
        super(MusicModel, self).__init__()
        self.sequence_size = sequence_size
        self.input_size = 1024
        self.dropout = dropout
        self.depth = len(levels_size)
        
        self.fc_cnn = nn.Linear(sequence_size, self.input_size)
        
        self.classification_blocks = nn.ModuleList()
        for level in range(1, self.depth + 1):
            current_input_size = self.input_size + sum(levels_size[f'level{i}'] for i in range(1, level))
            self.classification_blocks.append(
                ClassificationBlock(current_input_size, levels_size[f'level{level}'], dropout)
            )
        
        self.output_normalization = OutputNormalization()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_cnn(x))
        
        prev_output = x
        for block in self.classification_blocks:
            current_input = torch.cat((prev_output, x), dim=1)
            prev_output = block(current_input)
        
        normalized_output = self.output_normalization(prev_output)
        
        return normalized_output
