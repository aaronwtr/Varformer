import torch
import yaml

import torch.nn as nn
import preprocessing as preprocessing
import torch.nn.functional as F
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, BatchSampler, Sampler
from typing import Dict, List, Tuple, Iterator, Union, Iterable


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

    @staticmethod
    def _get_pos_neg_genes(_data, source):
        pos_data = getattr(_data, f"{source}_pos_data")
        neg_data = getattr(_data, f"{source}_neg_data")
        ids = getattr(_data, f"{source}_ids_all")
        pos_idx = pos_data.index.tolist()
        neg_idx = neg_data.index.tolist()
        pos_ensg = ids[pos_idx].tolist()
        neg_ensg = ids[neg_idx].tolist()
        return pos_ensg, neg_ensg, ids

    @staticmethod
    def _get_genes_per_source(_data, source):
        """
        Only works for gc and go modalities!
        """
        data = getattr(_data, f"{source}_data")
        all_test_ids = _data.all_test_ids
        data_ids = data.index.tolist()
        all_test_ensgs = all_test_ids[all_test_ids.index.isin(data_ids)].tolist()
        return list(dict.fromkeys(all_test_ensgs))

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
            if gene != 'labels':
                try:
                    pvc_data.data.pop(gene)
                    pvc_data.labels.pop(gene)
                except KeyError:
                    print(f"Gene {gene} not found in dataframe")

        for gene in list(pvc_data.labels.keys()):
            if gene not in list(pvc_data.data.keys()):
                pvc_data.labels.pop(gene)

        # homogenize test data
        for source in test_sources:
            gc_ensg = self._get_genes_per_source(gc_data, source)
            pvc_dict = getattr(pvc_data, f"{source}_data")
            pvc_ensg = list(pvc_dict.keys())

            common_genes = set(pvc_ensg).intersection(set(gc_ensg))

            for gene in common_genes:
                if gene not in pvc_ensg:
                    pvc_dict.pop(gene, None)

            setattr(pvc_data, f"{source}_data", pvc_dict)

            gc_all_ids = gc_data.all_test_ids
            gc_ids = gc_all_ids[gc_all_ids.isin(common_genes)].drop_duplicates().index.tolist()
            setattr(gc_data, f"{source}_data", getattr(gc_data, f"{source}_data").loc[gc_ids])
            setattr(go_data, f"{source}_data", getattr(go_data, f"{source}_data").loc[gc_ids])

        data = {
            "gc": gc_data,
            "go": go_data,
            "pvc": pvc_data
        }

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
                ensg_ids = preprocessor.ensg_ids
                combined_config = preprocessor.config
                test_labels = preprocessor.test_labels
            combined_features += preprocessor.num_features

        combined_genes.remove('labels')

        combined_test_data = {
            "pfam": {},
            "rcnt": {},
            "pharos": {}
        }
        combined_test_genes = {
            "pfam": {},
            "rcnt": {},
            "pharos": {}
        }

        for module, preprocessor in data_dict.items():
            if hasattr(preprocessor, 'pfam_data'):
                combined_test_data["pfam"][module] = preprocessor.pfam_data

                if isinstance(preprocessor.pfam_data, pd.DataFrame):
                    pfam_data_ids = preprocessor.pfam_data.index.tolist()
                    pfam_data_ids = ensg_ids.loc[pfam_data_ids].tolist()
                else:
                    pfam_data_ids = list(preprocessor.pfam_data.keys())

                # Ensure consistent order based on ensg_ids
                pfam_all_ids = [gene for gene in ensg_ids if gene in pfam_data_ids]
                combined_test_genes["pfam"][module] = pfam_all_ids

            if hasattr(preprocessor, 'rcnt_data'):
                combined_test_data["rcnt"][module] = preprocessor.rcnt_data

                if isinstance(preprocessor.rcnt_data, pd.DataFrame):
                    rcnt_data_ids = preprocessor.rcnt_data.index.tolist()
                    rcnt_data_ids = ensg_ids.loc[rcnt_data_ids].tolist()
                else:
                    rcnt_data_ids = list(preprocessor.rcnt_data.keys())

                # Ensure consistent order based on ensg_ids
                rcnt_all_ids = [gene for gene in ensg_ids if gene in rcnt_data_ids]
                combined_test_genes["rcnt"][module] = rcnt_all_ids

            if hasattr(preprocessor, 'pharos_data'):
                combined_test_data["pharos"][module] = preprocessor.pharos_data

                if isinstance(preprocessor.pharos_data, pd.DataFrame):
                    pharos_data_ids = preprocessor.pharos_data.index.tolist()
                    pharos_data_ids = ensg_ids.loc[pharos_data_ids].tolist()
                else:
                    pharos_data_ids = list(preprocessor.pharos_data.keys())

                # Ensure consistent order based on ensg_ids
                pharos_all_ids = [gene for gene in ensg_ids if gene in pharos_data_ids]
                combined_test_genes["pharos"][module] = pharos_all_ids

        return {
            "train": combined_train,
            "genes": list(combined_genes),
            "num_features": combined_features,
            "config": combined_config,
            "test_data": combined_test_data,
            "test_labels": test_labels,
            "test_genes": combined_test_genes
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


class MultiModalData(Dataset):
    def __init__(self, data, labels, gene_names, variant_data=None, max_variants=None, test_source=False):
        self.data = data
        self.labels = labels
        self.gene_names = gene_names
        self.variant_data = variant_data
        self.max_variants = max_variants
        self.test_source = test_source

        # if self.variant_data is not None:
        #     self.variant_features = {gene: self.variant_data['data'][gene] for gene in self.gene_names if
        #                              gene in self.variant_data['data']}
            # self.variant_labels = {gene: self.variant_data['labels'][gene] for gene in self.gene_names if
            #                       gene in self.variant_data['labels']}

    def __len__(self):
        if self.data is not None:
            return len(self.labels)
        elif self.variant_data is not None:
            return len(self.variant_features)
        else:
            return 0

    def __getitem__(self, index):
        if self.data is not None:
            gene_name = self.gene_names[index]
            if self.test_source is False:
                return (torch.tensor(self.data[gene_name], dtype=torch.float32),
                        torch.tensor(self.labels[gene_name], dtype=torch.float32))
            else:
                dataset = self.data[gene_name]
                labels = self.labels[gene_name]
                return (torch.tensor(dataset, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32),
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

    def padding(self, features):
        if features.size(0) < self.max_variants:
            padding = torch.zeros((self.max_variants - features.size(0)), dtype=features.dtype)
            features = torch.cat([features, padding], dim=0)
        elif features.size(0) > self.max_variants:
            features = features[:self.max_variants]
        return features


class SynchronizedMultiModalBatchSampler(BatchSampler):
    def __init__(self, dataset_dict: Dict[str, Dataset], batch_size: int, sampler: Union[Sampler[int], Iterable[int]],
                 shuffle: bool = True, drop_last: bool = False):
        """
        Custom batch sampler that ensures synchronized batching across multiple modalities.

        Args:
            dataset_dict: Dictionary of datasets for each modality
            batch_size: Size of each batch
            shuffle: Whether to shuffle the data
            drop_last: Whether to drop the last incomplete batch
        """
        super().__init__(sampler, batch_size, drop_last)
        self.dataset_dict = dataset_dict
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Verify all datasets have the same genes
        self._verify_gene_alignment()

        # Get common gene list (using any modality as they should all be the same)
        first_modality = next(iter(dataset_dict.values()))
        self.gene_names = first_modality.gene_names
        self.num_samples = len(self.gene_names)

    def _verify_gene_alignment(self):
        """Verify that all modalities have the same genes in the same order."""
        gene_lists = [list(dataset.gene_names) for dataset in self.dataset_dict.values()]
        if not all(genes == gene_lists[0] for genes in gene_lists):
            raise ValueError("All modalities must have the same list of genes in the same order!")

    def __iter__(self) -> Iterator[List[int]]:
        # Create index list
        indices = list(range(self.num_samples))

        if self.shuffle:
            # Use generator from PyTorch for reproducibility
            g = torch.Generator()
            g.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
            indices = torch.randperm(self.num_samples, generator=g).tolist()

        # Yield batches
        batch = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size


class MultiModalDataLoader:
    def __init__(self, datasets: Dict[str, Dataset], batch_size: int, shuffle: bool = True, drop_last: bool = False):
        """
        Custom DataLoader for handling multiple modalities.

        Args:
            datasets: Dictionary of datasets for each modality
            batch_size: Size of each batch
            shuffle: Whether to shuffle the data
            drop_last: Whether to drop the last incomplete batch
        """
        self.datasets = datasets
        self.batch_sampler = SynchronizedMultiModalBatchSampler(
            datasets, batch_size, shuffle, drop_last
        )

    def __iter__(self):
        for batch_indices in self.batch_sampler:
            batch = {}
            for modality, dataset in self.datasets.items():
                modality_batch = [dataset[i] for i in batch_indices]

                # Collate the batch
                if isinstance(modality_batch[0], dict):
                    # For variant data
                    batch[modality] = {}
                    for key in modality_batch[0].keys():
                        items = [item[key] for item in modality_batch]
                        if isinstance(items[0], torch.Tensor):
                            batch[modality][key] = torch.stack(items)
                        elif isinstance(items[0], (int, float, bool)):
                            batch[modality][key] = torch.tensor(items)
                        else:
                            batch[modality][key] = items
                else:
                    # For regular data
                    if modality == 'pvc':
                        print('break')
                    # check if item[0] in modality_batch is an numpy.ndarray
                    if isinstance(modality_batch[0][0], np.ndarray):
                        print('break')
                    features = torch.stack([item[0] for item in modality_batch])
                    labels = torch.stack([item[1] for item in modality_batch])
                    if len(modality_batch[0]) > 2:  # If test_source exists
                        test_source = modality_batch[0][2]  # Assuming test_source is same for batch
                        batch[modality] = (features, labels, test_source)
                    else:
                        batch[modality] = (features, labels)

            yield batch

    def __len__(self):
        return len(self.batch_sampler)
