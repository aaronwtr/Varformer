import torch
import torch.nn as nn


class PseudoLabelLoss(nn.Module):
    def __init__(self, output=None, lambd=0.1, pi=0.1, P=None, U=None, L=None, pseudo_labels=None):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.output = output
        self.lambd = lambd
        self.pi = pi
        self.P = P
        self.U = U
        self.L = L
        self.pseudo_labels = pseudo_labels

    @staticmethod
    def exp_sigmoid_loss(S, y):
        """
        Implement
        \$
            1 / |S| * \sum_{i \in S} 1 / (1 + e^{y_i * sigmoid(logits_i)})
        $\
        Note that S is the set with label y. and y is an integer in {-1, 1} corresponding to unlabeled and labeled
        respectively.
        """
        return 1 / len(S) * torch.sum(1 / (1 + torch.exp(y * S)))

    def forward(self):
        """
        :param preds: the predicted probabilities from the model
        :param P: the positive example indices
        :param U: the unlabeled example indices
        :param pseudo_labels: the pseudo labels in L
        :param lambd: loss mixing parameter
        :param pi: prior probability of positive samples (treated as hyperparam)
        """
        pseudo_preds = self.preds[self.L]

        L_L = self.bce_loss(pseudo_preds, self.pseudo_labels)

        L_P = self.exp_sigmoid_loss(self.preds[self.P], 1)
        L_U = self.exp_sigmoid_loss(self.preds[self.U], -1)

        L_PU = self.pi * L_P + torch.clamp(L_U - self.pi * L_P, min=0)

        loss = self.lambd * L_L + (1 - self.lambd) * L_PU

        return loss
