import os
import torch
from torch.utils.data import Dataset
import ast
import json
import pandas as pd
import numpy as np
import networkx as nx

from pathlib import Path
from tqdm.notebook import tqdm

from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import logging

# Configurar o logger
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Criar um logger
logger = logging.getLogger(__name__)


BUFFER_SIZE = 10

# Função para converter a string em lista de np.array
def convert_to_list_of_arrays(val):
    # Converte string para uma lista de strings
    list_str = ast.literal_eval(val)
    # Converte cada item da lista para um array NumPy
    return [np.array(item) for item in list_str]

class HMCDataset_old(Dataset):
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



# Nós que devem ser ignorados
to_skip = ['root', 'GO0003674', 'GO0005575', 'GO0008150']



def load_json(file_path):
    """
    Loads a JSON file and returns its content.
    
    Args:
        file_path (str): Path to the JSON file.
    
    Returns:
        dict: Parsed JSON data as a dictionary.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None


def load_dataset_paths(dataset_path):
    go_path = os.path.join(dataset_path, 'gene-ontology-annotated-datasets')
    fun_path = os.path.join(dataset_path, 'funcat-annotated-datasets')
    fma_path = os.path.join(dataset_path, 'fma_rock_electronic')

    datasets = {
        'cellcycle_FUN': (
            fun_path + '/cellcycle_FUN.train.csv',
            fun_path + '/cellcycle_FUN.valid.csv',
            fun_path + '/cellcycle_FUN.test.csv',
            fun_path + '/cellcycle_FUN-labels.json',
            fun_path + '/cellcycle_FUN.json'
        ),
        'derisi_FUN': (
            fun_path + '/derisi_FUN.train.csv',
            fun_path + '/derisi_FUN.valid.csv',
            fun_path + '/derisi_FUN.test.csv',
            fun_path + '/derisi_FUN-labels.json',
            fun_path + '/derisi_FUN.json'
        ),
        'eisen_FUN': (
            fun_path + '/eisen_FUN.train.csv',
            fun_path + '/eisen_FUN.valid.csv',
            fun_path + '/eisen_FUN.test.csv',
            fun_path + '/eisen_FUN-labels.json',
            fun_path + '/eisen_FUN.json'
        ),
        'expr_FUN': (
            fun_path + '/expr_FUN.train.csv',
            fun_path + '/expr_FUN.valid.csv',
            fun_path + '/expr_FUN.test.csv',
            fun_path + '/expr_FUN-labels.json',
            fun_path + '/expr_FUN.json'
        ),
        'gasch1_FUN': (
            fun_path + '/gasch1_FUN.train.csv',
            fun_path + '/gasch1_FUN.valid.csv',
            fun_path + '/gasch1_FUN.test.csv',
            fun_path + '/gasch1_FUN-labels.json',
            fun_path + '/gasch1_FUN.json'
        ),
        'gasch2_FUN': (
            fun_path + '/gasch2_FUN.train.csv',
            fun_path + '/gasch2_FUN.valid.csv',
            fun_path + '/gasch2_FUN.test.csv',
            fun_path + '/gasch2_FUN-labels.json',
            fun_path + '/gasch2_FUN.json'
        ),
        'seq_FUN': (
            fun_path + '/seq_FUN.train.csv',
            fun_path + '/seq_FUN.valid.csv',
            fun_path + '/seq_FUN.test.csv',
            fun_path + '/seq_FUN-labels.json',
            fun_path + '/seq_FUN.json'
        ),
        'spo_FUN': (
            fun_path + '/spo_FUN.train.csv',
            fun_path + '/spo_FUN.valid.csv',
            fun_path + '/spo_FUN.test.csv',
            fun_path + '/spo_FUN-labels.json',
            fun_path + '/spo_FUN.json'
        ),
        'cellcycle_GO': (
            go_path + '/cellcycle_GO.train.csv',
            go_path + '/cellcycle_GO.valid.csv',
            go_path + '/cellcycle_GO.test.csv',
            go_path + '/cellcycle_GO-labels.json',
            go_path + '/cellcycle_GO.json'
        ),
        'derisi_GO': (
            go_path + '/derisi_GO.train.csv',
            go_path + '/derisi_GO.valid.csv',
            go_path + '/derisi_GO.test.csv',
            go_path + '/derisi_GO-labels.json',
            go_path + '/derisi_GO.json'
        ),
        'eisen_GO': (
            go_path + '/eisen_GO.train.csv',
            go_path + '/eisen_GO.valid.csv',
            go_path + '/eisen_GO.test.csv',
            go_path + '/eisen_GO-labels.json',
            go_path + '/eisen_GO.json'
        ),
        'expr_GO': (
            go_path + '/expr_GO.train.csv',
            go_path + '/expr_GO.valid.csv',
            go_path + '/expr_GO.test.csv',
            go_path + '/expr_GO-labels.json',
            go_path + '/expr_GO.json'
        ),
        'gasch1_GO': (
            go_path + '/gasch1_GO.train.csv',
            go_path + '/gasch1_GO.valid.csv',
            go_path + '/gasch1_GO.test.csv',
            go_path + '/gasch1_GO-labels.json',
            go_path + '/gasch1_GO.json'
        ),
        'gasch2_GO': (
            go_path + '/gasch2_GO.train.csv',
            go_path + '/gasch2_GO.valid.csv',
            go_path + '/gasch2_GO.test.csv',
            go_path + '/gasch2_GO-labels.json',
            go_path + '/gasch2_GO.json'
        ),
        'seq_GO': (
            go_path + '/seq_GO.train.csv',
            go_path + '/seq_GO.valid.csv',
            go_path + '/seq_GO.test.csv',
            go_path + '/seq_GO-labels.json',
            go_path + '/seq_GO.json'
        ),
        'spo_GO': (
            go_path + '/spo_GO.train.csv',
            go_path + '/spo_GO.valid.csv',
            go_path + '/spo_GO.test.csv',
            go_path + '/spo_GO-labels.json',
            go_path + '/spo_GO.json'
        ),
        'fma_rock_electronic': (
            fma_path + '/train.csv',
            fma_path + '/valid.csv',
            fma_path + '/test.csv',
            fma_path + '/labels.json',
            fma_path + '/metadata.json'
        ),
    }
    
    return datasets


class HMCDataset(Dataset):
    def __init__(self, csv_file, nodes , nodes_idx, local_nodes_idx,  g_t , levels_size, is_fma = False, is_local=False, is_global=False):
        self.df = None
        self.features = None
        self.X = []
        self.X_cont = []
        self.X_bin = []
        self.Y = []
        self.Y_local = []
        self.csv_file = csv_file
        self.is_fma = is_fma
        self.is_local = is_local
        self.is_global = is_global
        self.nodes = nodes
        self.nodes_idx = nodes_idx
        self.local_nodes_idx = local_nodes_idx
        self.g_t = g_t
        self.levels_size = levels_size
        self.load_data()

    def __transform_labels(self):
        y_local_ = []
        y_ = []
        for labels in self.df.categories.values:
            if self.is_global:
                y_ = np.zeros(len(self.nodes))

            if self.is_local:
                sorted_keys = sorted(self.levels_size.keys())
                y_local_ = [np.zeros(self.levels_size.get(key)) for key in sorted_keys]

            for node in labels.split('@'):
                if self.is_global:
                    y_[[self.nodes_idx.get(a) for a in nx.ancestors(self.g_t, node)]] = 1
                    y_[self.nodes_idx[node]] = 1

                if self.is_local:
                    depth = nx.shortest_path_length(self.g_t, "root").get(node)
                    y_local_[depth][self.local_nodes_idx[depth].get(node)] = 1
                    for ancestor in nx.ancestors(self.g_t, node):
                        if ancestor != "root":
                            depth = nx.shortest_path_length(self.g_t, "root").get(ancestor)
                            y_local_[depth][self.local_nodes_idx[depth].get(ancestor)] = 1

            if self.is_global:
                self.Y.append(y_)

            if self.is_local:
                self.Y_local.append([np.stack(y) for y in y_local_])

        if self.is_global:
            self.Y = np.stack(self.Y)

    def transform_labels(self):
        self.__transform_labels()

    def parse_features(self):
        for features in self.features:
            cont_features = []
            bin_features = []
            for feature in features:
                # Se 'item' for uma lista, consideramos como binária
                if isinstance(feature, list):
                    # Achatar (flatten) essa sublista e colocar na bin_features
                    for f in feature:
                        bin_features.append(f)
                else:
                    # Se for float ou int, consideramos feature contínua
                    if feature is None or feature == 'None' or feature == 'nan' or feature == 'NaN' or type(
                            feature) is None:
                        cont_features.append(np.nan)
                    else:
                        cont_features.append(feature)
            self.X_cont.append(cont_features)
            self.X_bin.append(bin_features)

        self.X_cont = np.array(self.X_cont, dtype=float)
        ''''
        self.X_bin = np.array(self.X_bin, dtype=int)
        if self.X_bin.shape[0] > 0:
            self.X = np.concatenate([self.X_cont, self.X_bin], axis=1)
        else:
            self.X = self.X_cont
        '''
    def transform_features(self):
        self.features = self.df.features.apply(lambda x: ast.literal_eval(x)).tolist()
        self.parse_features()
        
    def load_data(self):
        """
        Load features and labels from CSV, and optionally a hierarchy graph from JSON.
        """
        if self.is_fma:
            # Load CSV for fma
            self.df = pd.read_csv(self.csv_file, delimiter="|")
        else:
            # Load CSV for others
            self.df = pd.read_csv(self.csv_file)
            self.transform_features()
        logger.info(f"Transforming labels")
        self.transform_labels()


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        features = torch.tensor(self.X[idx], dtype=torch.float32)
        labels = torch.tensor(self.Y[idx], dtype=torch.float32)
        return features, labels

class HMCDatasetManager:
    """
    Manages hierarchical multi-label datasets, including loading features (X), labels (Y),
    and optionally applying input scaling and hierarchical structure.

    Parameters:
    - dataset (tuple): Tuple containing paths to (train_csv, valid_csv, test_csv, labels_json, _).
    - output_path (str, optional): Path to store processed outputs. Default is 'data'.
    - device (str, optional): Computation device ('cpu' or 'cuda'). Default is 'cpu'.
    - is_local (bool, optional): Whether to use local hierarchy. Default is False.
    - is_global (bool, optional): Whether to use global hierarchy. Default is False.
    - input_scaler (bool, optional): Whether to apply input scaling (imputation + standardization). Default is True.

    """

    def __init__(self, dataset, output_path='data', device='cpu', is_local=False, is_global=False, input_scaler=True):
        # Extract dataset paths

        self.levels = {}
        self.levels_size = {}
        self.local_nodes_idx = {}
        self.categories = None
        self.roots = None
        self.R = None
        self.nodes = None
        self.nodes_idx = None
        self.g_t = None
        self.A = None

        train_csv, valid_csv, test_csv, labels_json, _ = dataset

        # Initialize attributes
        self.is_local = is_local
        self.is_global = is_global
        self.device = device

        # Infer dataset type
        self.is_go = any(keyword in train_csv for keyword in ['GO', 'go'])
        self.is_fma = any(keyword in train_csv for keyword in ['fma', 'FMA'])

        # Construct graph path
        self.g = nx.DiGraph()
        labels_json_name = Path(labels_json).name
        self.graph_path = os.path.join(output_path, labels_json_name.replace('-labels.json', '.graphml'))

        logger.info(f"Loading dataset from {train_csv} and {labels_json}")

        # Load hierarchical structure
        self.load_structure(labels_json)


        # Dataset configuration dictionary
        dataset_kwargs = {
            "nodes": self.nodes,
            "nodes_idx": self.nodes_idx,
            "local_nodes_idx": self.local_nodes_idx,
            "g_t": self.g_t,
            "levels_size": self.levels_size,
            "is_fma": self.is_fma,
            "is_local": self.is_local,
            "is_global": self.is_global,
        }

        # Load datasets
        self.train = HMCDataset(train_csv, **dataset_kwargs)
        self.val = HMCDataset(valid_csv, **dataset_kwargs)  # Corrigido para usar valid_csv
        self.test = HMCDataset(test_csv, **dataset_kwargs)

        # Apply scaling if needed
        if input_scaler:
            self.train, self.val = impute_scaler(self.train, self.val, device=device)

        # Ensure category labels exist before evaluation filtering
        self.to_eval = (
            [t not in to_skip for t in self.categories['labels']] if hasattr(self, 'categories') else []
        )

    def get_torch_dataset(self):
        return self.train, self.val, self.test

    def load_structure(self, labels_json):
        # Load labels JSON
        with open(labels_json, 'r') as f:
            self.categories = json.load(f)

        for cat in self.categories['labels']:
            terms = cat.split('.')
            if self.is_go:
                self.g.add_edge(terms[1], terms[0])
            else:
                if len(terms) == 1:
                    self.g.add_node(terms[0])
                    self.roots.append(terms[0])
                else:
                    for i in range(2, len(terms) + 1):
                        self.g.add_edge('.'.join(terms[:i]), '.'.join(terms[:i - 1]))

        self.nodes = sorted(self.g.nodes(), key=lambda x: (len(x.split('.')), x))
        self.nodes_idx = dict(zip(self.nodes, range(len(self.nodes))))
        self.g_t = self.g.reverse()

        ### Save networkx graph
        # Para salvar em formato GraphML
        nx.write_graphml(self.g, self.graph_path)

        self.A = nx.to_numpy_array(self.g, nodelist=self.nodes)
        if self.is_local:
            self.get_hierarchy_levels()

    def get_hierarchy_levels(self):
        """
        Retorna um dicionário com os nós agrupados por nível na hierarquia.
        """
        level_by_node = nx.shortest_path_length(self.g_t, "root")

        for node in self.g.nodes():
            depth = level_by_node.get(node)
            if depth not in self.levels:
                self.levels[depth] = []
            self.levels[depth].append(node)
        self.levels_size = {key: len(value) for key, value in self.levels.items()}
        print(self.levels_size)
        self.local_nodes_idx = {idx: dict(zip(level_nodes, range(len(level_nodes)))) for idx, level_nodes in
                                self.levels.items()}

    def compute_matrix_R(self):
        # Compute matrix of ancestors R
        # Given n classes, R is an (n x n) matrix where R_ij = 1 if class i is ancestor of class j
        R = np.zeros(self.A.shape)
        np.fill_diagonal(R, 1)
        g = nx.DiGraph(self.A)
        for i in range(len(self.A)):
            descendants = list(nx.descendants(g, i))
            if descendants:
                R[i, descendants] = 1
        R = torch.tensor(R)
        # Transpose to get the ancestors for each node
        R = R.transpose(1, 0)
        self.R = R.unsqueeze(0)


def impute_scaler(train, val, device: str = 'cpu'):
    """
    Impute missing values and apply standard scaling to continuous features.

    Parameters:
    - train: Dataset object containing X_cont (continuous features) and Y (labels).
    - val: Dataset object containing X_cont (continuous features) and Y (labels).
    - device (str, optional): Device to move labels (Y) to ('cpu' or 'cuda'). Default is 'cpu'.

    Returns:
    - Tuple (train, val): Updated train and validation datasets with imputed and scaled features.
    """

    scaler = preprocessing.StandardScaler().fit(np.concatenate((train.X_cont, val.X_cont)))
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit(np.concatenate((train.X_cont, val.X_cont)))
    val.X_count, val.Y = scaler.transform(imp_mean.transform(val.X_cont)), torch.tensor(val.Y).to(
        device)
    train.X_count, train.Y = scaler.transform(imp_mean.transform(train.X_cont)), torch.tensor(
        train.Y).to(device)

    return train, val


def initialize_dataset(
    name: str,
    dataset_path: str,
    output_path: str,
    device: str = "cpu",
    is_local: bool = False,
    is_global: bool = False,
    input_scaler: bool = True
) -> HMCDatasetManager:
    """
    Initialize and return an HMCDatasetManager for the specified dataset.

    Parameters:
    - name (str): Name of the dataset to load.
    - dataset_path (str): Path to the dataset directory.
    - output_path (str): Path to store output files.
    - device (str, optional): Device to be used ('cpu' or 'cuda'). Default is 'cpu'.
    - is_local (bool, optional): Whether to use local hierarchy. Default is False.
    - is_global (bool, optional): Whether to use global hierarchy. Default is False.

    Returns:
    - HMCDatasetManager: Initialized dataset manager.
    """
    # Load dataset paths
    datasets = load_dataset_paths(dataset_path)

    # Validate if the dataset exists
    if name not in datasets:
        raise ValueError(f"Dataset '{name}' not found in {dataset_path}. Available datasets: {list(datasets.keys())}")

    # Initialize dataset manager
    return HMCDatasetManager(
        datasets[name],
        output_path=output_path,
        device=device,
        is_local=is_local,
        is_global=is_global,
        input_scaler=input_scaler,
    )