import torch
from torch.utils.data import Dataset


class DrugTargetData(Dataset):
    def __init__(self, data, labels, gene_names, features):
        self.data = data
        self.labels = labels
        self.gene_names = gene_names
        self.features = features

        x = self.data
        y = self.labels
        # TODO: fix the bug: map everything to torch.bfloat16. Take care to map categorical features separately from
        #  numerical features.
        self.features = torch.tensor(x, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.int64)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

    def label_imbalance(self):
        return self.labels.sum() / len(self.labels)
