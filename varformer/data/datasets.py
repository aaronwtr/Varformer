"""DrugTargetData, VarformerDataset, MultiModalData — moved from src/dataloader.py (Phase 4A)."""
import torch
import numpy as np

from torch.utils.data import Dataset
from typing import Dict


class DrugTargetData(Dataset):
    def __init__(self, data, labels, gene_names, test_source=False):
        self.data = data
        self.labels = labels
        self.gene_names = gene_names

        self.test_source = test_source

        x = self.data
        y = self.labels

        self.features = torch.tensor(x, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, index):
        if self.test_source is False:
            return self.features[index], self.labels[index]
        else:
            return self.features[index], self.labels[index], self.test_source

    def __len__(self):
        return len(self.labels)

    def label_imbalance(self):
        return self.labels.sum() / len(self.labels)


class VarformerDataset(Dataset):
    def __init__(self, variant_data, max_variants: int, test_source=False):
        self.labels = variant_data['labels']
        self.variant_features = variant_data['data']
        self.max_variants = max_variants
        self.gene_names = list(self.variant_features.keys())
        self.test_source = test_source

    def __len__(self) -> int:
        return len(self.variant_features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        gene_name = self.gene_names[idx]
        variants_for_gene = self.variant_features[gene_name]
        gene_label = self.labels[gene_name]

        pat_feat = self.padding(variants_for_gene[:, 0])
        position = self.padding(variants_for_gene[:, 1])
        mut_feat = self.padding(variants_for_gene[:, 2])
        gene = self.padding(variants_for_gene[:, 3])

        # Create attention mask
        mask = torch.zeros(self.max_variants, dtype=torch.bool)
        mask[len(variants_for_gene):] = 1.0

        return {
            'pathogenicity': pat_feat.float(),
            'position': position.int(),
            'mutation': mut_feat.int(),
            'gene': gene.int(),
            'mask': mask.float(),
            'labels': gene_label,
            'test_source': self.test_source
        }

    def label_imbalance(self):
        if self.labels is not None:
            label_list = list(self.labels.values())
            return sum(label_list) / len(label_list)
        else:
            return 0

    def padding(self, features):
        if features.size(0) < self.max_variants:
            padding = torch.zeros((self.max_variants - features.size(0)), dtype=features.dtype)
            features = torch.cat([features, padding], dim=0)
        elif features.size(0) > self.max_variants:
            features = features[:self.max_variants]
        return features


class MultiModalData(Dataset):
    def __init__(self, data, labels, gene_names, dtype, variant_data=None, max_variants=None, test_source=False):
        self.data = data
        self.labels = labels
        self.gene_names = gene_names
        self.variant_data = variant_data
        self.max_variants = max_variants
        self.test_source = test_source
        self.torch_dtype = dtype
        if isinstance(self.torch_dtype, str):
            self.torch_dtype = torch.bfloat16 if dtype == 'bf16-mixed' else torch.float32
            torch.set_default_dtype(self.torch_dtype)


    def __len__(self):
        if self.data is not None:
            return len(self.labels)
        elif self.variant_data is not None:
            return len(self.variant_data['labels'])
        else:
            return 0

    def __getitem__(self, index):
        if self.data is not None:
            gene_name = self.gene_names[index]
            if self.test_source is False:
                return (torch.tensor(self.data[gene_name], dtype=self.torch_dtype),
                        torch.tensor(self.labels[gene_name], dtype=self.torch_dtype))
            else:
                dataset = self.data[gene_name]
                labels = self.labels[gene_name]
                return (torch.tensor(dataset, dtype=self.torch_dtype), torch.tensor(labels, dtype=self.torch_dtype),
                        self.test_source)
        elif self.variant_data is not None:
            gene_name = self.gene_names[index]
            variants_for_gene = self.variant_data['data'][gene_name]
            gene_label = self.variant_data['labels'][gene_name]

            pat_feat = self.padding(variants_for_gene[:, 0])
            position = self.padding(variants_for_gene[:, 1])
            mut_feat = self.padding(variants_for_gene[:, 2])
            gene = self.padding(variants_for_gene[:, 3])

            # Create attention mask
            mask = torch.zeros(self.max_variants, dtype=torch.bool)
            mask[len(variants_for_gene):] = 1.0

            return {
                'pathogenicity': pat_feat.float(),
                'position': position.int(),
                'mutation': mut_feat.int(),
                'gene': gene.int(),
                'mask': mask.float(),
                'labels': gene_label,
                'test_source': self.test_source,
                'gene_name': gene_name
            }

    def label_imbalance(self):
        return sum(list(self.labels.values())) / len(self.labels)

    def samples_per_class(self):
        labels = list(self.labels.values())
        return labels.count(0), labels.count(1)

    def padding(self, features):
        if features.size(0) < self.max_variants:
            padding = torch.zeros((self.max_variants - features.size(0)), dtype=features.dtype)
            features = torch.cat([features, padding], dim=0)
        elif features.size(0) > self.max_variants:
            features = features[:self.max_variants]
        return features
