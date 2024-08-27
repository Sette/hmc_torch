
import torch
import torch.nn as nn
import torch.nn.functional as F

def create_hierarchy_matrix(levels_size):
    hierarchy_matrix = []
    for i in range(len(levels_size) - 1):
        matrix = torch.zeros(levels_size[i+1], levels_size[i])
        for child in range(levels_size[i+1]):
            parent = child // 2  # Ajuste isso de acordo com a sua hierarquia específica
            matrix[child, parent] = 1
        hierarchy_matrix.append(matrix)
    return hierarchy_matrix


class WeightedMaskedBCELoss(nn.Module):
    def __init__(self, level_weights):
        super().__init__()
        self.level_weights = level_weights

    def forward(self, output, target, level):
        mask = (target != -1)
        loss = F.binary_cross_entropy_with_logits(output[mask], target[mask], reduction='mean')
        return loss * self.level_weights[level]


def hierarchical_loss(outputs, targets, level_weights, hierarchy_matrix):
    h_loss = 0
    for index, (output, target) in enumerate(zip(outputs, targets)):
        mask = (target != -1)
        level_loss = F.binary_cross_entropy_with_logits(output[mask], target[mask], reduction='mean')
        h_loss += level_loss * level_weights[index]

    # Adicionar penalidade para inconsistências hierárquicas
    for i in range(len(outputs) - 1):
        parent = torch.sigmoid(outputs[i])
        child = torch.sigmoid(outputs[i + 1])

        # Usar a matriz de hierarquia para mapear as classes filhas para as classes pai
        child_to_parent = torch.matmul(child, hierarchy_matrix[i].to(child.device))

        inconsistency = torch.max(child_to_parent - parent, torch.zeros_like(parent))
        h_loss += torch.mean(inconsistency) * level_weights[i]

    return h_loss


def show_local_losses(losses, set='train'):
    for i, loss in enumerate(losses):
        print(f'{set.capitalize()} Loss (Level {i + 1}): {loss:.4f}')


def show_global_loss(loss, set='train'):
    print(f'Global {set.capitalize()} Loss: {loss:.4f}')



class MaskedBCELoss(nn.Module):
    def __init__(self):
        super(MaskedBCELoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='none')  # Redução 'none' para manter a forma do tensor

    def forward(self, outputs, targets):
        losses = []
        for output, target in zip(outputs, targets):
            if len(target.shape) > 1:
                mask = target.sum(dim=1) > 0  # Dimensão 1 para targets 2D
            else:
                mask = target.sum() > 0  # Targets 1D ou outros casos

            if mask.any():
                loss = self.bce_loss(output, target)  # Calcula a perda sem redução
                masked_loss = loss[mask]  # Aplica a máscara
                losses.append(masked_loss.mean())  # Calcula a média da perda mascarada

        if len(losses) > 0:
            return torch.stack(losses).mean()  # Retorna um tensor e calcula a média
        else:
            return torch.tensor(0.0, requires_grad=True).to(outputs[0].device)  # Retorna uma perda zero se não houver perdas
