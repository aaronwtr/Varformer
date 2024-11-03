import torch
import yaml

import torch.nn as nn
import preprocessing as preprocessing
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset
from typing import Dict, List, Tuple


class ModuleDataProcessor:
    def __init__(self, gc, go, pvc, psc):
        assert any([gc, go, pvc, psc]), "Select at least one module to train the teacher model."
        self.gc = gc
        self.go = go
        self.pvc = pvc
        self.psc = psc

        with open("config.yml", 'r') as stream:
            self.config = yaml.safe_load(stream)

    def process(self):
        data = {'gc': None, 'go': None, 'pvc': None}
        if self.gc:
            data['gc'] = self.open_gc_data()
        if self.go:
            data['go'] = self.open_go_data(data['gc'])
        if self.pvc:
            data['pvc'] = self.open_pvc_data(data['gc'])
        return self.homogenize_data(data)

    def open_gc_data(self):
        gcp = preprocessing.GeneCharacterisationPreprocessor(config=self.config)
        print("Gene characterisation features preprocessed!\n")
        return gcp

    def open_go_data(self, gc_data):
        gop = preprocessing.GeneOntologyPreprocessor(config=self.config, gcp=gc_data)
        print("Gene ontology features preprocessed!\n")
        return gop

    def open_pvc_data(self, gc_data, tune=False):
        pvc = preprocessing.PopulationVariantPreprocessor(config=self.config, gcp=gc_data, tune=tune)
        print("Population variants preprocessed!\n")
        return pvc

    def homogenize_data(self, data):
        gc_data = data['gc']
        go_data = data['go']
        pvc_data = data['pvc']

        test_sources = ['pfam', 'rcnt', 'pharos']

        ensg_pvc = list(pvc_data.data.keys())
        ensg_gc = list(gc_data.ensg_ids.tolist())

        dropped_genes = list(set(ensg_gc) - set(ensg_pvc))
        dropped_gene_idx = gc_data.ensg_ids[gc_data.ensg_ids.isin(dropped_genes)].index.tolist()

        # homogenize train data
        data['gc'].data = data['gc'].data.drop(dropped_gene_idx, errors='ignore')
        data['go'].data = data['go'].data.drop(dropped_gene_idx, errors='ignore')
        gc_df_index = data['gc'].data.index.tolist()
        gc_ensg_ids = gc_data.ensg_ids[gc_df_index]
        dropped_genes = list(set(ensg_pvc) - set(gc_ensg_ids))
        for gene in dropped_genes:
            try:
                pvc_data.data.pop(gene)
            except KeyError:
                print(f"Gene {gene} not found in dataframe")

        # homogenize test data
        for source in test_sources:
            pvc_pos_dict = getattr(pvc_data, f"{source}_pos_dict")
            pvc_pos_ensg = list(pvc_pos_dict.keys())
            gc_pos_data = getattr(gc_data, f"{source}_pos_data")
            gc_neg_data = getattr(gc_data, f"{source}_neg_data")
            gc_ids = getattr(gc_data, f"{source}_ids_all")
            gc_pos_idx = gc_pos_data.index.tolist()
            gc_neg_idx = gc_neg_data.index.tolist()
            gc_pos_ensg = gc_ids[gc_pos_idx].tolist()
            gc_neg_ensg = gc_ids[gc_neg_idx].tolist()
            dropped_pos_genes = list(set(gc_pos_ensg) - set(pvc_pos_ensg))
            dropped_neg_genes = np.random.choice(gc_neg_ensg, len(dropped_pos_genes), replace=False)
            dropped_pos_idx = gc_ids[gc_ids.isin(dropped_pos_genes)].index.tolist()
            dropped_neg_idx = gc_ids[gc_ids.isin(dropped_neg_genes)].index.tolist()
            setattr(gc_data, f"{source}_data", getattr(gc_data, f"{source}_data").drop(dropped_pos_idx))
            setattr(gc_data, f"{source}_data", getattr(gc_data, f"{source}_data").drop(dropped_neg_idx))
            setattr(go_data, f"{source}_data", getattr(go_data, f"{source}_data").drop(dropped_pos_idx))
            setattr(go_data, f"{source}_data", getattr(go_data, f"{source}_data").drop(dropped_neg_idx))

        data['gc'].data.index = gc_ensg_ids
        data['go'].data.index = gc_ensg_ids

        return self.combine_modalities(data)

    @staticmethod
    def combine_modalities(data_dict):
        """
        Combines different modalities' data and their corresponding features
        """
        combined_train = {}
        combined_genes = set()
        combined_features = 0
        combined_config = {}

        for module, preprocessor in data_dict.items():
            combined_train[module] = preprocessor.data
            if module == 'pvc':
                combined_genes = list(set(preprocessor.data.keys()))
            if module == 'gc':
                combined_config = preprocessor.config
            combined_features += preprocessor.num_features

        combined_test_data = {
            "pfam": {},
            "rcnt": {},
            "pharos": {}
        }
        combined_test_genes = {
            "pfam": set(),
            "rcnt": set(),
            "pharos": set()
        }

        for module, preprocessor in data_dict.items():
            if hasattr(preprocessor, 'pfam_data'):
                combined_test_data["pfam"][module] = preprocessor.pfam_data
                combined_test_genes["pfam"].update(preprocessor.pfam_ids)

            if hasattr(preprocessor, 'rcnt_data'):
                combined_test_data["rcnt"][module] = preprocessor.rcnt_data
                combined_test_genes["rcnt"].update(preprocessor.rcnt_ids)

            if hasattr(preprocessor, 'pharos_data'):
                combined_test_data["pharos"][module] = preprocessor.pharos_data
                combined_test_genes["pharos"].update(preprocessor.pharos_ids)

        return {
            "train": combined_train,
            "genes": list(combined_genes),
            "num_features": combined_features,
            "config": combined_config,
            "test_data": combined_test_data,
            "test_genes": {k: list(v) for k, v in combined_test_genes.items()}
        }


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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
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


class DrugTargetVAEData(Dataset):
    def __init__(self, drug_target_data, reduct_dim, reduction_type="padding"):
        # Drug Target Data
        self.features = torch.tensor(drug_target_data['data'], dtype=torch.float32)
        self.labels = torch.tensor(drug_target_data['labels'],
                                   dtype=torch.float32) if 'labels' in drug_target_data else None
        self.gene_names = drug_target_data['gene_names']
        self.test_source = drug_target_data.get('test_source', False)

        # Variant Pathogenicity Data
        # self.variant_pathogenicities = [torch.tensor(v, dtype=torch.float32) for v in drug_target_data['data']]
        self.reduction_type = reduction_type
        self.reduct_dim = reduct_dim

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        label = self.labels[index] if self.labels is not None else torch.tensor(0.0)
        variant_data = self.features[index]
        variant_features = self.reduction(variant_data)

        if self.test_source is False:
            return variant_features, label
        else:
            return variant_features, label, self.test_source

    def reduction(self, x):
        if self.reduction_type == "padding":
            x = self.padding(x)
        elif self.reduction_type == "pooling":
            x = self.pooling(x)
        elif self.reduction_type == "None":
            return x
        else:
            raise ValueError("Invalid reduction type. Expected 'padding', 'pooling' or 'None'.")
        return x

    def padding(self, x):
        current_dimension = x.size(-1)
        if current_dimension == self.reduct_dim:
            return x
        elif current_dimension < self.reduct_dim:
            padding_left = (self.reduct_dim - current_dimension) // 2
            padding_right = self.reduct_dim - current_dimension - padding_left
            padded_x = F.pad(x, (padding_left, padding_right), value=0)
            return padded_x
        else:
            raise ValueError("Current dimension is already greater than the target dimension.")

    def pooling(self, x):
        current_dimension = x.size(-1)
        if current_dimension == self.reduct_dim:
            return x
        elif current_dimension > self.reduct_dim:
            pool = nn.AdaptiveAvgPool1d(self.reduct_dim)
            pooled_x = pool(x)
            return pooled_x
        else:
            raise ValueError("Current dimension is already smaller than the target dimension.")

    def label_imbalance(self):
        if self.labels is not None:
            return self.labels.sum() / len(self.labels)
        else:
            return 0


class VariantPathogenicityData(Dataset):
    def __init__(self, data_dict, reduct_dim, reduction_type="padding"):
        self.gene_names = list(data_dict.keys())
        self.variant_pathogenicities = list(data_dict.values())

        self.variant_pathogenicities = [torch.tensor(v, dtype=torch.float32) for v in self.variant_pathogenicities]

        self.reduction_type = reduction_type
        self.reduct_dim = reduct_dim

    def __len__(self):
        return len(self.gene_names)

    def __getitem__(self, idx):
        variant_pathogenicities = self.variant_pathogenicities[idx]
        reduct_dim = self.reduct_dim
        return variant_pathogenicities, reduct_dim

    def reduction(self, x):
        if self.reduction_type == "padding":
            x = self.padding(x)
        elif self.reduction_type == "pooling":
            x = self.pooling(x)
        else:
            raise ValueError("Invalid reduction type. Expected 'padding' or 'pooling'.")
        return x

    def padding(self, x):
        current_dimension = x.size(-1)
        if current_dimension == self.reduct_dim:
            return x
        elif current_dimension < self.reduct_dim:
            # Calculate the required padding on both sides
            padding_left = (self.reduct_dim - current_dimension) // 2
            padding_right = self.reduct_dim - current_dimension - padding_left

            # Pad the tensor
            padded_x = F.pad(x, (padding_left, padding_right), value=0)
            return padded_x
        else:
            raise ValueError("Current dimension is already greater than the target dimension.")

    def pooling(self, x):
        current_dimension = x.size(-1)

        if current_dimension == self.reduct_dim:
            return x
        elif current_dimension > self.reduct_dim:
            pool = nn.AdaptiveAvgPool1d(self.reduct_dim)
            pooled_x = pool(x)
            return pooled_x
        else:
            raise ValueError("Current dimension is already smaller than the target dimension.")
