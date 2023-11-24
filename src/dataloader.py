import torch
from torch.utils.data import Dataset


class DrugTargetData(Dataset):
    def __init__(self, data, labels, gene_names, features, batch_size: int = 32, num_workers: int = 7):
        self.data = data
        self.labels = labels
        self.gene_names = gene_names
        self.features = features
        self.batch_size = batch_size
        self.num_workers = num_workers

        x = self.data
        y = self.labels
        self.features = torch.tensor(x, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.int64)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

    def label_imbalance(self):
        return self.labels.sum() / len(self.labels)
