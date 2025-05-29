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
    def __init__(
        self,
        input_shape,
        hidden_size,
        output_size,
        nb_layers,
        dropout_rate=0.5,
    ):
        super(BuildClassification, self).__init__()
        layers = []
        layers.append(nn.Linear(input_shape, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        for _ in range(nb_layers - 1):  # Add additional hidden layers
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_size, output_size))
        layers.append(nn.Sigmoid())  # Sigmoid for binary classification

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)


class HMCLocalModel(nn.Module):
    def __init__(
        self,
        levels_size,
        input_size=None,
        hidden_size=None,
        num_layers=None,
        dropout=None,
        active_levels=None,
    ):
        super(HMCLocalModel, self).__init__()
        if not input_size:
            print("input_size is None, error in HMCLocalClassificationModel")
            raise ValueError("input_size is None")
        if not levels_size:
            print("levels_size is None, error in HMCLocalClassificationModel")
            raise ValueError("levels_size is None")
        if not isinstance(levels_size, dict):
            if active_levels is None:
                print("levels_size is not a dict, error in HMCLocalClassificationModel")
                raise ValueError("levels_size is not a dict")

        self.input_size = input_size
        self.levels_size = levels_size
        self.mum_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.levels = nn.ModuleList()
        self.active_levels = active_levels

        if active_levels is not None:
            self.model = BuildClassification(
                input_shape=input_size,
                hidden_size=hidden_size,
                output_size=levels_size,
                nb_layers=num_layers,
                dropout_rate=dropout,
            )
        else:
            self.max_depth = len(levels_size)
            for index, level_size in enumerate(levels_size.values()):
                self.levels.append(
                    BuildClassification(
                        input_shape=input_size,
                        hidden_size=hidden_size[index],
                        output_size=level_size,
                        nb_layers=num_layers[index],
                        dropout_rate=dropout[index],
                    )
                )

    def forward(self, x):
        outputs = []
        if self.active_levels is not None:
            output = self.model(x)
            return output
        else:
            for idx, level in enumerate(self.levels):
                local_output = level(x)
                outputs.append(local_output)
            return outputs
