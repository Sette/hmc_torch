
import torch
import numpy as np
import networkx as nx
import logging
from hmc.utils import __load_json__
from hmc.dataset.datasets.gofun import get_dataset_paths

from hmc.dataset.datasets.gofun import to_skip

from hmc.dataset import  HMCDatasetCsv, HMCDatasetTorch, HMCDatasetArff

# Configurar o logger
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Criar um logger
logger = logging.getLogger(__name__)


class HMCDatasetManager:
    """
    Manages hierarchical multi-label datasets, including loading features (X), labels (Y),
    and optionally applying input scaling and hierarchical structure.

    Parameters:
    - dataset (tuple): Tuple containing paths to (train_csv, valid_csv, test_csv, labels_json, _).
    - output_path (str, optional): Path to store processed outputs. Default is 'data'.
    - device (str, optional): Computation device ('cpu' or 'cuda'). Default is 'cpu'.
    - is_local (bool, optional): Whether to use local_classifier hierarchy. Default is False.
    - is_global (bool, optional): Whether to use global hierarchy. Default is False.
    - input_scaler (bool, optional): Whether to apply input scaling (imputation + standardization). Default is True.

    """

    def __init__(self, dataset, dataset_type='csv', device='cpu', is_global=False):
        # Extract dataset paths
        self.test, self.train, self.valid, self.to_eval = None, None, None, None
        self.levels, self.levels_size,  self.nodes_idx, self.local_nodes_idx = {}, {}, {}, {}
        self.labels, self.roots, self.nodes, self.g_t, self.A = [], [], [], [], []
        self.to_skip = to_skip
        # Initialize attributes
        self.is_global = is_global
        self.device = device

        # Construct graph path
        self.g = nx.DiGraph()
        # train_csv_name = Path(self.train_file).name
        # self.graph_path = os.path.join(output_path, train_csv_name.replace('-.csv', '.graphml'))

        if dataset_type == 'arff':
            self.is_go, self.train_file, self.valid_file, self.test_file = dataset
        else:
            self.train_file, self.valid_file, self.test_file, self.labels_file = dataset
            # Infer dataset type
            self.is_go = any(keyword in self.train_file for keyword in ['GO', 'go'])
            self.is_fma = any(keyword in self.train_file for keyword in ['fma', 'FMA'])
            # Load hierarchical structure
            self.load_structure_from_json(self.labels_file)

        logger.info(f"Loading dataset from {self.train_file}")

        if dataset_type == 'csv':
            self.load_csv_data()
            self.to_eval = [t not in self.to_skip for t in self.nodes]
        elif dataset_type == 'torch':
            self.load_torch_data()
            self.to_eval = [t not in self.to_skip for t in self.nodes]
        elif dataset_type == 'arff':
            self.load_arff_data()
            self.to_eval = self.train.to_eval


        # Ensure category labels exist before evaluation filtering
        #self.to_eval = [t not in self.to_skip for t in self.nodes]


    def load_structure_from_json(self, labels_json):
        # Load labels JSON
        self.labels = __load_json__(labels_json)
        for cat in self.labels['labels']:
            terms = cat.split('/')
            if self.is_go:
                self.g.add_edge(terms[1], terms[0])
            else:
                if len(terms) == 1:
                    self.g.add_edge(terms[0], 'root')
                else:
                    for i in range(2, len(terms) + 1):
                        self.g.add_edge('.'.join(terms[:i]), '.'.join(terms[:i - 1]))

        self.nodes = sorted(self.g.nodes(), key=lambda x: (nx.shortest_path_length(self.g, x, 'root'), x) if self.is_go else (len(x.split('.')), x))
        self.nodes_idx = dict(zip(self.nodes, range(len(self.nodes))))
        self.g_t = self.g.reverse()

        ### Save networkx graph
        # Para salvar em formato GraphML
        #nx.write_graphml(self.g, self.graph_path)

        self.A = nx.to_numpy_array(self.g, nodelist=self.nodes)
        if not self.is_global:
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
        # Compute matrix of ancestors R, named matrix_r
        # Given n classes, R is an (n x n) matrix where R_ij = 1 if class i is ancestor of class j
        matrix_r = np.zeros(self.A.shape)
        np.fill_diagonal(matrix_r, 1)
        g = nx.DiGraph(self.A)
        for i in range(len(self.A)):
            descendants = list(nx.descendants(g, i))
            if descendants:
                matrix_r[i, descendants] = 1
        matrix_r = torch.tensor(matrix_r)
        # Transpose to get the ancestors for each node
        matrix_r = matrix_r.transpose(1, 0)
        matrix_r = matrix_r.unsqueeze(0)
        return matrix_r

    def transform_labels(self, dataset_labels):
        y_local_ = []
        y_ = []
        Y = []
        Y_local = []
        for labels in dataset_labels:
            if self.is_global:
                y_ = np.zeros(len(self.nodes))
            else:
                sorted_keys = sorted(self.levels_size.keys())
                y_local_ = [np.zeros(self.levels_size.get(key)) for key in sorted_keys]
            for node in labels.split('@'):
                if self.is_global:
                    y_[[self.nodes_idx.get(a) for a in nx.ancestors(self.g_t, node)]] = 1
                    y_[self.nodes_idx[node]] = 1

                if not self.is_global:
                    depth = nx.shortest_path_length(self.g_t, "root").get(node)
                    y_local_[depth][self.local_nodes_idx[depth].get(node)] = 1
                    for ancestor in nx.ancestors(self.g_t, node):
                        if ancestor != "root":
                            depth = nx.shortest_path_length(self.g_t, "root").get(ancestor)
                            y_local_[depth][self.local_nodes_idx[depth].get(ancestor)] = 1

            if self.is_global:
                Y.append(y_)
            else:
                Y_local.append([np.stack(y) for y in y_local_])
                return Y_local
        if self.is_global:
            Y = np.stack(Y)
            return Y

    def load_csv_data(self):
        """
        Load features and labels from CSV, and optionally a hierarchy graph from JSON.
        """
        self.train = HMCDatasetCsv(self.train_file, is_go=self.is_go)
        self.valid = HMCDatasetCsv(self.valid_file, is_go=self.is_go)
        self.test = HMCDatasetCsv(self.test_file, is_go=self.is_go)

        dataset_labels = self.train.df.labels.values
        logger.info(f"Transforming train labels")
        self.train.set_y(self.transform_labels(dataset_labels))

        dataset_labels = self.valid.df.labels.values
        logger.info(f"Transforming valid labels")
        self.valid.set_y(self.transform_labels(dataset_labels))

        dataset_labels = self.test.df.labels.values
        logger.info(f"Transforming test labels")
        self.test.set_y(self.transform_labels(dataset_labels))

    def load_torch_data(self):
        self.train = HMCDatasetTorch(self.train_file)
        self.valid = HMCDatasetTorch(self.valid_file)
        self.test = HMCDatasetTorch(self.test_file)

        dataset_labels = self.train.Y
        logger.info(f"Transforming train labels")
        self.train.set_y(self.transform_labels(dataset_labels))

        dataset_labels = self.valid.Y
        logger.info(f"Transforming valid labels")
        self.valid.set_y(self.transform_labels(dataset_labels))

        dataset_labels = self.test.Y
        logger.info(f"Transforming test labels")
        self.test.set_y(self.transform_labels(dataset_labels))

    def load_arff_data(self):
        self.train = HMCDatasetArff(self.train_file, is_go=self.is_go)
        self.valid = HMCDatasetArff(self.valid_file, is_go=self.is_go)
        self.test = HMCDatasetArff(self.test_file, is_go=self.is_go)
        self.A = self.train.A

    def get_datasets(self):
        return self.train, self.valid, self.test


def initialize_dataset_experiments(
    name: str,
    device: str = "cpu",
    dataset_type='torch',
    is_global: bool = False,
) -> HMCDatasetManager:
    """
    Initialize and return an HMCDatasetManager for the specified dataset.

    Parameters:
    - name (str): Name of the dataset to load.
    - output_path (str): Path to store output files.
    - device (str, optional): Device to be used ('cpu' or 'cuda'). Default is 'cpu'.
    - is_local (bool, optional): Whether to use local_classifier hierarchy. Default is False.
    - is_global (bool, optional): Whether to use global hierarchy. Default is False.

    Returns:
    - HMCDatasetManager: Initialized dataset manager.
    """
    # Load dataset paths
    datasets = get_dataset_paths(dataset_type=dataset_type)


    # Validate if the dataset exists
    if name not in datasets:
        raise ValueError(f"Dataset '{name}' not found in experiments datasets. Available datasets: {list(datasets.keys())}")

    # Initialize dataset manager
    return HMCDatasetManager(
        datasets[name],
        dataset_type=dataset_type,
        device=device,
        is_global=is_global,
    )


