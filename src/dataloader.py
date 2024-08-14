import torch
import yaml

import torch.nn as nn
import src.preprocessing as preprocessing
import torch.nn.functional as F

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


class ShardedVarformerDataset(Dataset):
    def __init__(self, data, shard_size=512):
        self.shard_size = shard_size
        self.gene_data = data['data']
        self.labels = data['labels']
        self.gene_names = list(self.gene_data.keys())
        self.sharded_data = []

        for gene, features in self.gene_data.items():
            num_variants = features.size(0)
            num_shards = (num_variants + shard_size - 1) // shard_size  # Ceiling division

            for i in range(num_shards):
                start = i * shard_size
                end = min((i + 1) * shard_size, num_variants)

                self.sharded_data.append({
                    'gene_id': gene,
                    'shard_id': i,
                    'pathogenicity': features[start:end, 0],
                    'position': features[start:end, 1],
                    'mutation': features[start:end, 2],
                    'total_shards': num_shards
                })

    def __len__(self):
        return len(self.sharded_data)

    def __getitem__(self, idx):
        shard = self.sharded_data[idx]
        num_variants = len(shard['pathogenicity'])

        # TODO: ADD LOGIC FOR PADDING HERE
        pathogenicity = F.pad(shard['pathogenicity'].clone().detach(), (0, self.shard_size - num_variants))
        position = F.pad(shard['position'].clone().detach(), (0, self.shard_size - num_variants))
        mutation = F.pad(shard['mutation'].clone().detach(), (0, self.shard_size - num_variants))

        # Create mask (1 for actual data, 0 for padding)
        mask = torch.cat([torch.ones(num_variants), torch.zeros(self.shard_size - num_variants)])

        return {
            'pathogenicity': pathogenicity.float(),
            'position': position.int(),
            'mutation': mutation.int(),
            'mask': mask.float(),
            'gene_id': shard['gene_id'],
            'shard_id': shard['shard_id'],
            'total_shards': shard['total_shards'],
            'labels': self.labels[shard['gene_id']]
        }

    def label_imbalance(self):
        if self.labels is not None:
            label_list = list(self.labels.values())
            return sum(label_list) / len(label_list)
        else:
            return 0


class VarformerDataset(Dataset):
    def __init__(self, variant_data, max_variants: int):
        self.genes = variant_data['data']
        self.labels = variant_data['labels']
        self.max_variants = max_variants
        self.gene_names = list(self.genes.keys())
        self.test_source = variant_data.get('test_source', False)

    def __len__(self) -> int:
        return len(self.genes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        gene_name = self.gene_names[idx]
        variant_features = self.genes[gene_name]
        label = self.labels[gene_name]

        pat_feat = self.padding(variant_features[:, 0])
        pos_feat = self.padding(variant_features[:, 1])
        mut_feat = self.padding(variant_features[:, 2])

        features = {
            'pathogenicity': pat_feat,
            'position': pos_feat,
            'mutation': mut_feat
        }

        # Create attention mask
        mask = torch.zeros(self.max_variants, dtype=torch.bool)
        mask[len(self.genes[gene_name]):] = True

        return features, mask, label, gene_name

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
        self.labels = torch.tensor(drug_target_data['labels'], dtype=torch.float32) if 'labels' in drug_target_data else None
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
