import numpy as np


def local_to_global_predictions(local_labels, local_nodes_idx, nodes_idx):
    n_samples = local_labels[0].shape[0]
    n_global_labels = len(nodes_idx)
    global_preds = np.zeros((n_samples, n_global_labels))
    sorted_levels = sorted(local_nodes_idx.keys())
    local_nodes_reverse = {
        level: {v: k for k, v in local_nodes_idx[level].items()}
        for level in sorted_levels
    }
    # print(f"Local nodes idx: {local_nodes_idx}")
    # print(f"Local nodes reverse: {local_nodes_reverse}")

    print(f"Exemplos: {n_samples}")
    # print(f"Shape local_preds: {len(local_labels)}")
    # print(f"Local nodes idx: {local_nodes_reverse}")

    # Etapa 1: montar node_names ativados por exemplo
    activated_nodes_by_example = [[] for _ in range(n_samples)]

    for level_index, level in enumerate(sorted_levels):
        level_preds = local_labels[
            level_index
        ]  # shape: [n_samples, n_classes_at_level]
        for idx_example, label in enumerate(level_preds):
            local_indices = np.where(label == 1)[0]  # aceita floats ou binários
            for local_idx in local_indices:
                node_name = local_nodes_reverse[level].get(local_idx)
                if node_name:
                    activated_nodes_by_example[idx_example].append(node_name)
                else:
                    print(
                        f"[WARN] Índice local {local_idx} não encontrado no nível {level}"
                    )

    # print(f"Node names ativados por exemplo: {activated_nodes_by_example[0]}")
    global_indices = []
    for node in activated_nodes_by_example[0]:
        # print(f"Node names ativados: {node}")
        if "/" in node:
            node = node.replace("/", ".")
        global_indices.append(nodes_idx.get(node))
    print(global_indices)
    # Etapa 2: converter node_names para índices globais
    for idx_example, node_names in enumerate(activated_nodes_by_example):
        for node_name in node_names:
            key = node_name.replace("/", ".")
            if key in nodes_idx:
                global_idx = nodes_idx[key]
                global_preds[idx_example][global_idx] = 1
            else:
                print(f"[WARN] Node '{key}' não encontrado em nodes_idx")

    return global_preds
