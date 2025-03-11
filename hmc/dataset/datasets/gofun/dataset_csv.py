

class csv_data():
    def __init__(self, csv_file, labels_json, is_go):
        self.csv_file = csv_file
        self.labels = __load_json__(labels_json)['labels']
        self.X, self.Y, self.A, self.terms, self.g = parse_csv(csv_file=self.csv_file, labels=self.labels , is_go=is_go)
        self.to_eval = [t not in to_skip for t in self.terms]


def parse_csv(csv_file, labels, is_go=False):
    df = pd.read_csv(csv_file, sep='|')
    Y = []
    #X = df['features'].tolist()
    df['features'] = df['features'].apply(json.loads)
    X = df['features'].tolist()
    #X = np.array([json.loads(x) for x in df['features']])
    g = nx.DiGraph()
    for branch in labels:
        terms = branch.split('.')
        if is_go:
            g.add_edge(terms[1], terms[0])
        else:
            if len(terms) == 1:
                g.add_edge(terms[0], 'root')
            else:
                for i in range(2, len(terms) + 1):
                    g.add_edge('.'.join(terms[:i]), '.'.join(terms[:i - 1]))

    nodes = sorted(g.nodes(), key=lambda x: (nx.shortest_path_length(g, x, 'root'), x) if is_go else (
        len(x.split('.')), x))
    nodes_idx = dict(zip(nodes, range(len(nodes))))
    g_t = g.reverse()
    data_labels = df['labels'].values.tolist()
    for data in data_labels:
        y_ = np.zeros(len(nodes))
        for t in data.split('@'):
            y_[[nodes_idx.get(a) for a in nx.ancestors(g_t, t.replace('/', '.'))]] = 1
            y_[nodes_idx[t.replace('/', '.')]] = 1
        Y.append(y_)
    X = np.array(X)
    Y = np.stack(Y)

    return X, Y, np.array(nx.to_numpy_array(g, nodelist=nodes)), nodes, g

