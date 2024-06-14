from torch.utils.data import Dataset
import torch

# Função parse_single_music adaptada
def parse_single_music(data, labels):
    track_id, categories, music = data
    max_depth = len(categories[0])
    data_dict = {}
    for level in range(1, max_depth + 1):
        level_labels = []
        for cat in categories:
            if cat[level-1] != "":
                label = labels[f'label_{level}'][cat[level-1]]
                if label not in level_labels:
                    level_labels.append(label)
            else:
                if len(level_labels) == 0:
                    level_labels.append(-1)
        data_dict[f'label{level}'] = level_labels

    data_dict['features'] = music
    data_dict['track_id'] = track_id

    return data_dict



# Classe personalizada do Dataset
class MusicDataset(Dataset):
    def __init__(self, dataframe, labels):
        self.data = dataframe
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        parsed_data = parse_single_music(data, self.labels)
        features = torch.tensor(parsed_data['features'], dtype=torch.float32)
        labels = {key: torch.tensor(value, dtype=torch.long) for key, value in parsed_data.items() if key.startswith('label')}
        track_id = torch.tensor(parsed_data['track_id'], dtype=torch.long)
        return features, labels, track_id