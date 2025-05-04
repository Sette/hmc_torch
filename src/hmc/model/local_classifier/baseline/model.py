import torch.nn as nn


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


class BuildClassification(nn.Module):
    def __init__(self, input_shape, hidden_size, output_size, dropout_rate=0.5):
        super(BuildClassification, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_shape, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid(),  # Sigmoid for binary classification
        )

    def forward(self, x):
        return self.classifier(x)


class HMCLocalClassificationModel(nn.Module):
    def __init__(self, levels_size, input_size=1280, hidden_size=640, num_layers=2, dropout=0.5):
        super(HMCLocalClassificationModel, self).__init__()
        self.input_size = input_size
        self.levels_size = levels_size
        self.mum_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.levels = nn.ModuleList()
        for level_size in levels_size.values():
            self.levels.append(BuildClassification(input_size, hidden_size, level_size))

    def forward(self, x):
        outputs = []
        for i, level in enumerate(self.levels):
            local_output = level(x)
            outputs.append(local_output)
        return outputs

        # def forward(self, x):
        #     outputs = []
        #     current_input = x
        #     current_output = current_input
        #     for i, level in enumerate(self.levels):
        #         if i != 0:
        #             current_input = torch.cat((current_output.detach(), x), dim=1)
        #         local_output = level(current_input)
        #         outputs.append(local_output)
        #         current_output = self.output_normalization[i](local_output)
        #     return outputs
        """
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
                # for level_output, _ in zip(outputs_per_level, self.thresholds):
                    # Aplica o threshold para converter em saída binária (0 ou 1)
                    # Sbinary_output = (level_output >= threshold).float()  #  threshold
                    # Converte para NumPy e armazena
                    # batch_predictions.append(binary_output.cpu().detach().numpy())
                    # batch_predictions.append(level_output.cpu().detach().numpy())
                    # SArmazena as previsões do batch atual para todos os níveis
            #predictions.append(batch_predictions)
            #output_list = [level_targets for level_targets in zip(*predictions)]
            #output_list = transform_predictions(output_list)

        #df_test['predictions'] = output_list
        return predictions
    """
