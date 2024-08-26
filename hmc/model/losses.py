

import torch.nn as nn

def show_local_losses(local_losses, set='train'):
    for level, local_loss in enumerate(local_losses):
        print(f'{set} Loss {local_loss:.4f} for level {level}')

def show_global_loss(global_loss, epoch, set='train'):
    print(f'Epoch {epoch}/{args.epochs}, Global {set} Loss: {global_loss:.4f}, Global {set} Loss: {global_loss:.4f}')

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
