import os
import pandas as pd
import ast
import logging

os.environ["DATA_FOLDER"] = "./"

# Configurar o logger
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

# Criar um logger
logger = logging.getLogger(__name__)

BUFFER_SIZE = 10

# Skip the root nodes
to_skip = ["root", "GO0003674", "GO0005575", "GO0008150"]


class HMCDatasetCsv:
    def __init__(self, data):
        """
        Inicializa o dataset.
        :param data: Estrutura de dados carregada do .pt
        """
        self.df = pd.read_csv(data, delimiter="|")
        self.X = []
        self.Y = []
        self.transform_features()

    def transform_features(self):
        self.df.features = self.df.features.apply(lambda x: ast.literal_eval(x))
        self.X = self.df.features.tolist()

    def set_y(self, y):
        self.Y = y
