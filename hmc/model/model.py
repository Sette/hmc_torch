import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import pandas as pd
import os

import torch.nn.functional as F
from torch.utils.data import DataLoader
from hmc.dataset import HMCDataset
from hmc.model.metrics import custom_thresholds, custom_dropouts, custom_lrs, custom_optimizers

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

class OutputNormalization(nn.Module):
    def __init__(self):
        super(OutputNormalization, self).__init__()

    def forward(self, x):
        # Obtém o índice do maior valor em cada linha (axis=1)
        max_indices = torch.argmax(x, dim=1)
        # Converte para one-hot encoding
        one_hot_output = F.one_hot(max_indices, num_classes=x.size(1))
        return one_hot_output.to(x.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape


class BuildClassification(nn.Module):
    def __init__(self, input_shape, hidden_size, output_size):
        super(BuildClassification, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_shape, hidden_size),
            nn.Linear(hidden_size, output_size)
        )
        self._initialize_weights()

    def forward(self, x):
        return self.classifier(x)
    
    def _initialize_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, a=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

class HMCLocalClassificationModel(nn.Module):
    def __init__(self, levels_size, input_size=1280, hidden_size=640, dropouts=None, thresholds=None, optimizers=None , lrs=None):
        super().__init__()
        self.input_size = input_size
        self.levels_size = levels_size
        self.dropouts = dropouts if dropouts is not None else custom_dropouts(len(levels_size))
        self.thresholds = thresholds if thresholds is not None else custom_thresholds(len(levels_size))
        self.lrs = lrs if lrs is not None else custom_lrs(len(levels_size))
        self.optimizers = optimizers if optimizers is not None else custom_optimizers(len(levels_size))
        self.levels = nn.ModuleList()
        self.output_normalization = nn.ModuleList()
        next_size = 0
        for level_size in levels_size:
            self.levels.append(BuildClassification(input_size + next_size, hidden_size, level_size))
            self.output_normalization.append(OutputNormalization())
            next_size = level_size
        
        
    def forward(self, x):
        outputs = []
        current_input = x
        current_output = current_input
        for i, level in enumerate(self.levels):
            if i != 0:
                current_input = torch.cat((current_output.detach(), x), dim=1)
            local_output = level(current_input)
            outputs.append(local_output)
            current_output = self.output_normalization[i](local_output)
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
                # Aplicando a sigmoid em cada tensor da lista de outputs
                prob_per_level = [F.sigmoid(output) for output in outputs_per_level]

                #print(outputs_per_level)
                levels_pred = {}
                for level, pred in enumerate(prob_per_level, start=1):
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
    

def get_constr_out(x, R):
    """
    Given the network output x and a constraint matrix R, 
    returns the modified output according to the hierarchical constraints in R.
    """
    # Convert x to double precision
    c_out = x.double()
    
    # Add a dimension to c_out: from (N, D) to (N, 1, D)
    # N: batch size, D: dimensionality of the output
    c_out = c_out.unsqueeze(1)
    
    # Expand c_out to match the shape of R:
    # If R is (C, C), c_out becomes (N, C, C)
    c_out = c_out.expand(len(x), R.shape[1], R.shape[1])
    
    # Expand R similarly to (N, C, C)
    R_batch = R.expand(len(x), R.shape[1], R.shape[1])
    
    # Element-wise multiplication of R_batch by c_out. 
    # This produces a (N, C, C) tensor.
    # torch.max(...) is taken along dimension=2, resulting in (N, C).
    # This extracts the maximum along the last dimension, effectively applying the hierarchical constraints.
    final_out, _ = torch.max(R_batch * c_out, dim=2)
    
    return final_out

class ConstrainedFFNNGlobalModel(nn.Module):
    """ C-HMCNN(h) model - during training it returns the not-constrained output that is then passed to MCLoss """
    def __init__(self, input_dim, hidden_dim, output_dim, hyperparams, R):
        super(ConstrainedFFNNGlobalModel, self).__init__()
        
        self.nb_layers = hyperparams['num_layers']
        self.R = R
        
        fc = []
        for i in range(self.nb_layers):
            if i == 0:
                fc.append(nn.Linear(input_dim, hidden_dim))
            elif i == self.nb_layers-1:
                fc.append(nn.Linear(hidden_dim, output_dim))
            else:
                fc.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc = nn.ModuleList(fc)
        
        self.drop = nn.Dropout(hyperparams['dropout'])
        
        
        self.sigmoid = nn.Sigmoid()
        if hyperparams['non_lin'] == 'tanh':
            self.f = nn.Tanh()
        else:
            self.f = nn.ReLU()
        
    def forward(self, x):
        for i in range(self.nb_layers):
            if i == self.nb_layers-1:
                x = self.sigmoid(self.fc[i](x))
            else:
                x = self.f(self.fc[i](x))
                x = self.drop(x)
        if self.training:
            constrained_out = x
        else:
            constrained_out = get_constr_out(x, self.R)
        return constrained_out