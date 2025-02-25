import os
import torch
from torch.utils.data import Dataset
import ast
import json
import pandas as pd
import numpy as np
import networkx as nx

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


class HMCDataset:
    def __init__(self, csv_file, labels_json, output_path = 'data'):
        """
        Initializes the dataset, loading features (X), labels (Y), and optionally the hierarchy graph.
        """
        self.g = None
        self.is_go = False
        self.is_fma = False
        self.nodes_idx = None
        self.categories = None
        self.nodes = None
        self.g_t = None
        self.df = None
        self.A = None
        self.max_len = 0
        self.X_cont = []
        self.Y = []
        labels_json_name = labels_json.split('/')[-1]
        self.graph_path = os.path.join(output_path, labels_json_name.replace('-labels.json', '.graphml'))

        if 'GO' in csv_file:
            self.is_go = True
        if 'fma' in csv_file:
            self.is_fma = True

        self.load_structure(labels_json)

        self.load_data(
            csv_file
        )
        self.to_eval = [t not in to_skip for t in self.categories['labels']]


    def load_structure(self, labels_json):
        # Load labels JSON
        with open(labels_json, 'r') as f:
            self.categories = json.load(f)

        self.g = nx.DiGraph()
        for cat in self.categories['labels']:
            terms = cat.split('.')

            cat_len = len(terms)
            logger.info(f'Category: {cat} - Length: {cat_len}')
            if cat_len > self.max_len:
                self.max_len = cat_len
                logger.info(f'New max length: {self.max_len}')

            if self.is_go:
                self.g.add_edge(terms[1], terms[0])
            else:
                if len(terms) == 1:
                    logger.info(f'Adding a root node: {terms[0]}')
                    self.g.add_node(terms[0])
                else:
                    logger.info(f'Adding edge: {".".join(terms[:2])} -> {".".join(terms[:1])}')
                    for i in range(2, len(terms) + 1):
                        self.g.add_edge('.'.join(terms[:i]), '.'.join(terms[:i - 1]))

            self.nodes = sorted(self.g.nodes(), key=lambda x: (len(x.split('.')), x))
            self.nodes_idx = dict(zip(self.nodes, range(len(self.nodes))))
            self.g_t = self.g.reverse()

        ### Save networkx graph
        # Para salvar em formato GraphML
        nx.write_graphml(self.g, self.graph_path)

        self.A = nx.to_numpy_array(self.g, nodelist=self.nodes)

    def get_hierarchy_levels(self):
        """
        Retorna um dicionário com os nós agrupados por nível na hierarquia.
        """
        levels = {}
        for node in self.g.nodes():
            depth = len(node.split('.'))
            if depth not in levels:
                levels[depth] = []
            levels[depth].append(node)
        return levels


    def transform_labels(self):
        for labels in self.df.categories.values:
            y_ = np.zeros(len(self.nodes))
            for t in labels.split('@'):
                y_[[self.nodes_idx.get(a) for a in nx.ancestors(self.g_t, t)]] = 1
                y_[self.nodes_idx[t]] = 1
            self.Y.append(y_)
        self.Y = np.stack(self.Y)

    def parse_features(self):
        self.X_cont = []
        self.X_bin = []
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
                    if feature is None or feature == 'None' or feature == 'nan' or feature == 'NaN' or type(feature) is None:
                        cont_features.append(np.nan)
                    else:
                        cont_features.append(feature)
            self.X_cont.append(cont_features)
            self.X_bin.append(bin_features)
        
        self.X_cont = np.array(self.X_cont, dtype=float)
        self.X_bin = np.array(self.X_bin, dtype=int)


    def transform_features(self):
        self.features = self.df.features.apply(lambda x : ast.literal_eval(x)).tolist()


    def load_data(self, csv_file):
        """
        Load features and labels from CSV, and optionally a hierarchy graph from JSON.
        """
        if self.is_fma:
            # Load CSV for fma
            self.df = pd.read_csv(csv_file, delimiter="|")
        else:
            # Load CSV for others
            self.df = pd.read_csv(csv_file)
            self.transform_features()
        self.transform_labels()

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







##### OLD

class GOFUNDataset:
    def __init__(self, csv_file, labels_json, output_path='data'):
        """
        Initializes the dataset, loading features (X), labels (Y), and optionally the hierarchy graph.
        """
        self.g = None
        self.nodes_idx = None
        self.g_t = None
        self.df = None
        self.X_cont = []
        self.is_go = False
        self.is_fma = False
        if 'GO' in csv_file:
            self.is_go = True

        labels_json_name = labels_json.split('/')[-1]
        self.graph_path = os.path.join(output_path, labels_json_name.replace('-labels.json', '.graphml'))
        self.columns_path = labels_json.replace('-labels', '')
        self.load_structure(labels_json)
        with open(self.columns_path, 'r') as f:
            self.columns = json.load(f)

        self.load_data(
            csv_file
        )
        self.to_eval = [t not in to_skip for t in self.categories['labels']]

    def load_structure(self, labels_json):
        # Load labels JSON
        with open(labels_json, 'r') as f:
            self.categories = json.load(f)

        self.g = nx.DiGraph()

        for cat in self.categories['labels']:
            terms = cat.split('.')
            if self.is_go:
                self.g.add_edge(terms[1], terms[0])
            else:
                if len(terms) == 1:
                    self.g.add_edge(terms[0], 'root')
                else:
                    for i in range(2, len(terms) + 1):
                        self.g.add_edge('.'.join(terms[:i]), '.'.join(terms[:i - 1]))

        ### Save networkx graph
        # Para salvar em formato GraphML
        nx.write_graphml(self.g, self.graph_path)

        self.nodes = sorted(self.g.nodes(),
                            key=lambda x: (nx.shortest_path_length(self.g, x, 'root'), x) if self.is_go else (
                            len(x.split('.')), x)
                            )
        self.nodes_idx = dict(zip(self.nodes, range(len(self.nodes))))
        self.g_t = self.g.reverse()
        self.A = nx.to_numpy_array(self.g, nodelist=self.nodes)

    def transform_labels(self):
        self.Y = []
        y_ = np.zeros(len(self.nodes))
        for labels in self.df.categories.values:
            for t in labels.split('@'):
                y_[[self.nodes_idx.get(a) for a in nx.ancestors(self.g_t, t)]] = 1
                y_[self.nodes_idx[t]] = 1
            self.Y.append(y_)
        self.Y = np.stack(self.Y)

    def parse_features(self):
        self.X_cont = []
        self.X_bin = []
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
        self.X_bin = np.array(self.X_bin, dtype=int)

    def transform_features(self):
        self.features = self.df.features.apply(lambda x: ast.literal_eval(x)).tolist()
        self.parse_features()

    def load_data(self, csv_file):
        """
        Load features and labels from CSV, and optionally a hierarchy graph from JSON.
        """
        # Load CSV
        self.df = pd.read_csv(csv_file)
        self.transform_features()
        self.transform_labels()

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


def impute_scaler(train, val, device='cuda'):
    scaler = preprocessing.StandardScaler().fit(np.concatenate((train.X_cont, val.X_cont)))
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit(np.concatenate((train.X_cont, val.X_cont)))
    val.X_count, val.Y = scaler.transform(imp_mean.transform(val.X_cont)), torch.tensor(val.Y).to(
        device)
    train.X_count, train.Y = scaler.transform(imp_mean.transform(train.X_cont)), torch.tensor(
        train.Y).to(device)

    return train, val

def initialize_dataset(name, dataset_path, output_path):
    """
    Initialize train, validation, and test datasets.
    """
    datasets = load_dataset_paths(dataset_path)
    train_csv, valid_csv, test_csv, labels_json, _ = datasets[name]


    train_data = HMCDataset(train_csv, labels_json, output_path)
    val_data = HMCDataset(valid_csv, labels_json, output_path)
    test_data = HMCDataset(test_csv, labels_json, output_path)
    return train_data, val_data, test_data
