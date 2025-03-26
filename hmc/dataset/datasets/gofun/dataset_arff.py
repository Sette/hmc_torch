
from hmc.dataset.datasets.gofun import to_skip
import keras

import networkx as nx
import pandas as pd
import numpy as np
from itertools import chain
from collections import defaultdict

def get_depth_by_root(g_t, t, roots):
    for root in roots:
        depth = nx.shortest_path_length(g_t, t, root)
        if depth is not None:
            return depth
    return None

class HMCDatasetArff():
    def __init__(self, arff_file, is_go):
        self.arff_file = arff_file
        self.X, self.Y, self.Y_local, self.A, self.terms, self.g, self.levels, self.levels_size, self.nodes_idx, self.local_nodes_idx, self.max_depth = parse_arff(arff_file=arff_file, is_go=is_go)
        self.to_eval = [t not in to_skip for t in self.terms]
        r_, c_ = np.where(np.isnan(self.X))
        m = np.nanmean(self.X, axis=0)
        for i, j in zip(r_, c_):
            self.X[i,j] = m[j]


def parse_arff(arff_file, is_go=False):
    global to_skip
    with open(arff_file) as f:
        read_data = False
        X = []
        Y = []
        Y_local = []
        levels_size = defaultdict(int)
        levels = defaultdict(list)
        g = nx.DiGraph()
        feature_types = []
        d = []
        cats_lens = []
        all_terms = []
        for num_line, l in enumerate(f):
            if l.startswith('@ATTRIBUTE'):
                if l.startswith('@ATTRIBUTE class'):
                    h = l.split('hierarchical')[1].strip()
                    for branch in h.split(','):
                        terms = branch.split('/')
                        all_terms.append(branch)
                        level = branch.count('/')  # Count the number of '.' to determine the level
                        levels[level].append(branch)
                        if is_go:
                            g.add_edge(terms[1], terms[0])
                        else:
                            if len(terms) == 1:
                                g.add_edge(terms[0], 'root')
                            else:
                                for i in range(2, len(terms) + 1):
                                    g.add_edge('.'.join(terms[:i]), '.'.join(terms[:i - 1]))
                    levels_size = {key: len(set(value)) for key, value in levels.items()}
                    print(f'Levels size: {levels_size}')
                    #print(f'Levels: {levels}')
                    nodes = sorted(g.nodes(), key=lambda x: (nx.shortest_path_length(g, x, 'root'), x) if is_go else (
                        len(x.split('.')), x))
                    nodes_idx = dict(zip(nodes, range(len(nodes))))
                    g_t = g.reverse()
                    max_depth = len(levels_size)
                    local_nodes_idx = {idx: dict(zip(level_nodes, range(len(level_nodes)))) for idx, level_nodes in
                                            levels.items()}
                else:
                    _, f_name, f_type = l.split()

                    if f_type == 'numeric' or f_type == 'NUMERIC':
                        d.append([])
                        cats_lens.append(1)
                        feature_types.append(lambda x, i: [float(x)] if x != '?' else [np.nan])

                    else:
                        cats = f_type[1:-1].split(',')
                        cats_lens.append(len(cats))
                        d.append({key: keras.utils.to_categorical(i, len(cats)).tolist() for i, key in enumerate(cats)})
                        feature_types.append(lambda x, i: d[i].get(x, [0.0] * cats_lens[i]))
            elif l.startswith('@DATA'):
                read_data = True
            elif read_data:
                y_ = np.zeros(len(nodes))
                sorted_keys = sorted(levels_size.keys())
                y_local_ = [np.zeros(levels_size.get(key)) for key in sorted_keys]
                d_line = l.split('%')[0].strip().split(',')
                lab = d_line[len(feature_types)].strip()

                X.append(list(chain(*[feature_types[i](x, i) for i, x in enumerate(d_line[:len(feature_types)])])))

                for t in lab.split('@'):
                    y_[[nodes_idx.get(a) for a in nx.ancestors(g_t, t.replace('/', '.'))]] = 1
                    y_[nodes_idx[t.replace('/', '.')]] = 1

                    ### Local labels
                    #depth =  get_depth_by_root(g_t, t, to_skip)
                    depth = nx.shortest_path_length(g, t.replace('/', '.'), "root")
                    assert depth != None

                    y_local_[depth-1][local_nodes_idx[depth-1].get(t.replace('/', '.'))] = 1
                    for ancestor in nx.ancestors(g_t, t.replace('/', '.')):
                        if ancestor not in  to_skip:
                            depth = nx.shortest_path_length(g, ancestor , "root")
                            y_local_[depth - 1][local_nodes_idx[depth - 1].get(t.replace('/', '.'))] = 1
                            y_local_[depth-1][local_nodes_idx[depth-1].get(ancestor)] = 1

                Y.append(y_)
                Y_local.append([np.stack(y) for y in y_local_])
        X = np.array(X)
        Y = np.stack(Y)

        return X, Y, Y_local , np.array(nx.to_numpy_array(g, nodelist=nodes)), nodes, g, levels, levels_size, nodes_idx , local_nodes_idx, max_depth

