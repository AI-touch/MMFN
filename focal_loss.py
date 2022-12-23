import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        log_pt = -ce_loss
        pt = torch.exp(log_pt)
        weights = (1 - pt) ** self.gamma
        fl = weights * ce_loss

        if self.reduction == 'sum':
            fl = fl.sum()
        elif self.reduction == 'mean':
            fl = fl.mean()
        else:
            raise ValueError(f"reduction '{self.reduction}' is not valid")
        return fl

# a1 = torch.from_numpy(np.array([0.8,0.2]))
# a2 = torch.from_numpy(np.array([0.6,0.4]))
# y = torch.from_numpy(np.array([1.0,0.0])).long()
#
# loss = F.cross_entropy(a1, y, reduction='mean')
# print(loss)