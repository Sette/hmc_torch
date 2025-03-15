
from hmc.dataset.datasets.gofun import to_skip

import networkx as nx
import pandas as pd
import numpy as np
import ast
import json
import os

def load_and_concat_csv_files(directory, sep='|'):
    """
    Loads all CSV files from a given directory and concatenates them into a single DataFrame.
    :param directory: Path to the directory containing CSV files.
    :return: A single concatenated DataFrame.
    """
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    dataframes = []

    for file in csv_files:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path, sep=sep)
        # Convert 'features' column from string representation of lists to actual lists
        if 'features' in df.columns:
            print('convertendo string em array')
            df['features'] = df['features'].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x)

        dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True) if dataframes else None

class HMCDatasetCsv():
    def __init__(self, csv_path, is_go):
        self.df = pd.DataFrame()
        self.X, self.Y = None, None
        self.is_go = is_go
        self.csv_path = csv_path
        self.parse_csv()

    def set_y(self, y):
        self.Y = np.stack(y)
        self.X = np.array(self.X)

    def parse_csv(self):
        #self.df = pd.read_csv(self.csv_path, sep='|')
        self.df = load_and_concat_csv_files(self.csv_path, sep='|')
        #X = df['features'].tolist()
        #self.df['features'] = self.df['features'].apply(json.loads)
        self.X = self.df['features'].tolist()
        self.Y = self.df['labels'].tolist()


        #X = np.array([json.loads(x) for x in df['features']])



