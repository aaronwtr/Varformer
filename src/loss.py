import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class PseudoLabelLoss(nn.Module):
    def __init__(self, output=None, lambd=0.1, pi=0.1, P=None, U=None, L=None, L_reset=None, labels=None,
                 pseudo_labels=None):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.output = output
        self.lambd = lambd
        self.pi = pi
        self.P = P
        self.U = U
        self.L = L
        self.L_reset = L_reset
        self.labels = labels
        self.pseudo_labels = pseudo_labels

    @staticmethod
    def exp_sigmoid_loss(S, y):
        """
        Implement
        \$
            1 / |S| * \sum_{i \in S} 1 / (1 + e^{y_i * sigmoid(logits_i)})
        $\
        Note that S is the set with label y. and y is an integer in {0, 1} corresponding to unlabeled and labeled
        respectively.
        """
        return 1 / len(S) * torch.sum(1 / (1 + torch.exp(y * S)))

    def forward(self, outputs, P, U, L, L_reset, labels, pseudo_labels):
        self.output = outputs
        self.P = P
        self.U = U
        self.L = L
        self.L_reset = L_reset
        self.labels = labels
        self.pseudo_labels = pseudo_labels

        if len(self.L) == 0:
            L_L = 0
        else:
            pseudo_preds = torch.gather(self.output, 0, self.L_reset)
            L_L = self.bce_loss(pseudo_preds, pseudo_labels)

        if len(self.P) == 0:
            L_P = 0
        else:
            L_P = self.exp_sigmoid_loss(torch.gather(self.output, 0, self.P), self.labels[self.P])
        L_U = self.exp_sigmoid_loss(torch.gather(self.output, 0, self.U), self.labels[self.U])

        L_PU = self.pi * L_P + torch.clamp(L_U - self.pi * L_P, min=0)

        loss = self.lambd * L_L + (1 - self.lambd) * L_PU

        return loss


class WarmupLinearScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_iters: int, total_iters: int, last_epoch: int = -1):
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        super(WarmupLinearScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_iters for base_lr in self.base_lrs]
        else:
            # Constant learning rate after warmup
            return [base_lr for base_lr in self.base_lrs]

