import torch
import torch.nn as nn
import torch.nn.functional as F

class OutputNormalization(nn.Module):
    def __init__(self):
        super(OutputNormalization, self).__init__()

    def forward(self, x):
        # Obtemos a classe com a maior probabilidade
        one_hot_encoded = torch.zeros_like(x).scatter_(1, x.argmax(dim=1, keepdim=True), 1.0)
        return one_hot_encoded

class ClassificationLayer(nn.Module):
    def __init__(self, input_shape, size, dropout):
        super(ClassificationLayer, self).__init__()
        self.fc1 = nn.Linear(input_shape, 1024)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(512, 256)
        self.batch_norm = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(dropout)
        self.fc4 = nn.Linear(256, size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.batch_norm(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

class MusicModel(nn.Module):
    def __init__(self, levels_size, sequence_size=1280, dropout=0.6):
        super(MusicModel, self).__init__()
        self.sequence_size = sequence_size
        self.classification_layers = nn.ModuleList()
        self.output_normalization = OutputNormalization()
        self.batch_norm = nn.BatchNorm1d(sequence_size)
        
        for level, size in levels_size.items():
            if level == 'level1':
                self.classification_layers.append(ClassificationLayer(sequence_size, size, dropout))
            else:
                self.classification_layers.append(ClassificationLayer(sequence_size * 2, size, dropout))

    def forward(self, x):
        outputs = []
        current_output = x
        
        for i, classification_layer in enumerate(self.classification_layers):
            if i > 0:
                normalized_output = self.batch_norm(self.output_normalization(current_output))
                normalized_output = normalized_output.repeat(1, self.sequence_size // normalized_output.shape[1])
                current_input = torch.cat([normalized_output, x], dim=1)
            else:
                current_input = x

            current_output = classification_layer(current_input)
            outputs.append(current_output)
        
        return outputs

