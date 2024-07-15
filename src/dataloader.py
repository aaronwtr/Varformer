import torch
import yaml

import torch.nn as nn
import preprocessing as preprocessing
import torch.nn.functional as F
import pandas as pd

from torch.utils.data import Dataset


class ModuleDataProcessor:
    def __init__(self, gc, go, pvc, psc):
        assert any([gc, go, pvc, psc]), "Select at least one module to train the teacher model."
        self.gc = gc
        self.go = go
        self.pvc = pvc
        self.psc = psc

        with open("src/config.yml", 'r') as stream:
            self.config = yaml.safe_load(stream)

    def process(self):
        data = {'gc': None, 'go': None, 'pvc': None}
        if self.gc:
            data['gc'] = self.open_gc_data()
        if self.go:
            data['go'] = self.open_go_data(data['gc'])
        if self.pvc:
            data['pvc'] = self.open_pvc_data(data['gc'])
        return data

    def open_gc_data(self):
        gcp = preprocessing.GeneCharacterisationPreprocessor(config=self.config)
        print("Gene characterisation features preprocessed!\n")
        return gcp

    def open_go_data(self, gc_data):
        gop = preprocessing.GeneOntologyPreprocessor(config=self.config, gcp=gc_data)
        print("Gene ontology features preprocessed!\n")
        return gop

    def open_pvc_data(self, gc_data):
        pvc = preprocessing.PopulationVariantPreprocessor(config=self.config, gcp=gc_data)
        print("Population variants preprocessed!\n")

        # pathcty_embds = vgep.pathogenicity_embeddings
        #
        # pthcty_df = pd.DataFrame.from_dict(pathcty_embds, orient='index')
        #
        # pthcty_df = pthcty_df.reset_index()
        # pthcty_df = pthcty_df.rename(columns={'index': 'ENSG'})
        #
        # pthcty_df = pthcty_df.rename(columns={i: f"pathogenicity_{i}" for i in range(0, pthcty_df.shape[1] - 1)})

        return pvc


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


class DrugTargetVAEData(Dataset):
    def __init__(self, drug_target_data, reduct_dim, reduction_type="padding"):
        # Drug Target Data
        self.features = torch.tensor(drug_target_data['data'], dtype=torch.float32)
        self.labels = torch.tensor(drug_target_data['labels'], dtype=torch.float32) if 'labels' in drug_target_data else None
        self.gene_names = drug_target_data['gene_names']
        self.test_source = drug_target_data.get('test_source', False)

        # Variant Pathogenicity Data
        self.variant_pathogenicities = [torch.tensor(v, dtype=torch.float32) for v in drug_target_data['data']]
        self.reduction_type = reduction_type
        self.reduct_dim = reduct_dim

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        label = self.labels[index] if self.labels is not None else torch.tensor(0.0)

        variant_data = self.variant_pathogenicities[index]
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
        else:
            raise ValueError("Invalid reduction type. Expected 'padding' or 'pooling'.")
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
