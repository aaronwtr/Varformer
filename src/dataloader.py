import torch
from torch.utils.data import Dataset


class DrugTargetData(Dataset):
    def __init__(self, data):
        self.data = data

        self.gene_names = self.data.iloc[:, 0].values

        x = self.data.iloc[:, 1:-1].values
        y = self.data.iloc[:, -1].values
        self.features = torch.tensor(x, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.int64)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.labels)
