import torch
from torch.utils.data import Dataset


class DrugTargetData(Dataset):
    def __init__(self, data, labels, gene_names):
        self.data = data
        self.labels = labels
        self.gene_names = gene_names

        x = self.data
        y = self.labels

        self.features = torch.tensor(x, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

    def label_imbalance(self):
        return self.labels.sum() / len(self.labels)


class VariantPathogenicityData(Dataset):
    def __init__(self, data_dict):
        self.gene_names = list(data_dict.keys())
        self.variant_pathogenicities = list(data_dict.values())

        self.variant_pathogenicities = [torch.tensor(v, dtype=torch.float32) for v in self.variant_pathogenicities]

    def __len__(self):
        return len(self.gene_names)

    def __getitem__(self, idx):
        gene_name = self.gene_names[idx]
        variant_pathogenicities = self.variant_pathogenicities[idx]
        return gene_name, variant_pathogenicities
