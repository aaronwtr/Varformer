import torch
import yaml
import random

import torch.nn as nn
import preprocessing as preprocessing
import torch.nn.functional as F
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, BatchSampler, Sampler
from typing import Dict, List, Tuple, Iterator, Union, Iterable
from tabulate import tabulate
from math import ceil


class ModuleDataProcessor:
    def __init__(self, gc, go, pvc, psc, config=None):
        assert any([gc, go, pvc, psc]), "Select at least one module to train the teacher model."
        self.gc = gc
        self.go = go
        self.pvc = pvc
        self.psc = psc

        if config is None:
            with open("cluster_config.yml", 'r') as stream:
                self.config = yaml.safe_load(stream)
        else:
            self.config = config

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

    def open_pvc_data(self, gc_data):
        pvc = preprocessing.PopulationVariantPreprocessor(config=self.config, gcp=gc_data)
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
        return list(set(all_test_ids[all_test_ids.index.isin(data_ids)].index.tolist()))

    @staticmethod
    def _clean_test_data(combined_test_data):
        """
        Remove target columns and labels from test data.
        Handles missing columns/keys gracefully without errors.
        """
        # Handle DataFrame columns for GC and GO
        for source in ['pfam', 'rcnt', 'pharos']:
            for modality in ['gc', 'go']:
                try:
                    if modality in combined_test_data[source] and isinstance(combined_test_data[source][modality],
                                                                             pd.DataFrame):
                        if 'target' in combined_test_data[source][modality].columns:
                            combined_test_data[source][modality].drop(columns=['target'], inplace=True)
                except (KeyError, AttributeError):
                    pass

            # Handle dictionary labels for PVC
            try:
                if 'pvc' in combined_test_data[source] and isinstance(combined_test_data[source]['pvc'], dict):
                    combined_test_data[source]['pvc'].pop('labels', None)  # None ensures no error if key doesn't exist
            except KeyError:
                pass

        return combined_test_data

    def homogenize_data(self, data):
        gc_data = data['gc']
        go_data = data['go']
        pvc_data = data['pvc']

        # Get gene sets from each modality
        ensg_pvc = set(pvc_data.data.keys())
        if 'labels' in ensg_pvc:  # Remove 'labels' key if present
            ensg_pvc.remove('labels')

        ensg_gc = set(gc_data.data.index.tolist())
        ensg_go = set(go_data.data.index.tolist())

        # Find the intersection of all three sets
        common_genes = ensg_gc.intersection(ensg_pvc).intersection(ensg_go)

        print(f"Original gene counts: GC={len(ensg_gc)}, GO={len(ensg_go)}, PVC={len(ensg_pvc)}")
        print(f"Common genes across all modalities: {len(common_genes)}")

        # Keep only common genes in each modality
        data['gc'].data = data['gc'].data[data['gc'].data.index.isin(common_genes)]
        data['go'].data = data['go'].data[data['go'].data.index.isin(common_genes)]

        data['go'].data = data['go'].data.loc[:, (data['go'].data != 0).any(axis=0)]

        # For PVC data which is in dictionary format
        for gene in list(pvc_data.data.keys()):
            if gene != 'labels' and gene not in common_genes:
                pvc_data.data.pop(gene)
                if gene in pvc_data.labels:
                    pvc_data.labels.pop(gene)

        # Verify all modalities now have the same number of genes
        gc_count = len(data['gc'].data)
        go_count = len(data['go'].data)
        pvc_count = len(pvc_data.data) - (1 if 'labels' in pvc_data.data else 0)

        assert gc_count == go_count == pvc_count, "Gene counts don't match across modalities"

        gc_data = data['gc']
        go_data = data['go']
        pvc_data = data['pvc']

        test_data_result = self.get_test_data(gc_data, go_data, pvc_data)

        # TODO: properly process the result from the test_data function based on if we are in eval or inference mode
        combined_test_data = test_data_result["test_data"]
        combined_test_genes = test_data_result["all_test_ids"]
        test_labels = test_data_result["test_labels"]
        test_labels_per_source = test_data_result["test_genes"]
        class_prior = test_data_result["class_prior"]

        feature_data = None
        config = None
        combined_train = {}
        combined_features = 0
        for module, preprocessor in data.items():
            feature_data = preprocessor.data
            combined_features += preprocessor.num_features
            if isinstance(feature_data, pd.DataFrame):
                feature_data = feature_data[~feature_data.index.isin(combined_test_genes)]
                config = preprocessor.config
            elif isinstance(feature_data, dict):
                feature_data = {gene: feature_data[gene] for gene in feature_data if gene not in combined_test_genes}
                feature_data.pop('labels')
            else:
                raise ValueError("Unsupported data type for feature_data. Should be DataFrame for GC and GO. Should"
                                 "be dict for PVC.")

            combined_train[module] = feature_data

        assert config is not None, "Config should be set!"
        assert class_prior is not None, "Class prior should be set!"

        if isinstance(feature_data, pd.DataFrame):
            combined_genes = set(feature_data.index.tolist())
        elif isinstance(feature_data, dict):
            combined_genes = set(feature_data.keys())
        else:
            raise ValueError("Unsupported data type for feature_data. Should be DataFrame for GC and GO. Should"
                             "be dict for PVC.")

        labels = dict(zip(combined_train['gc'].index, combined_train['gc']['target'])) if 'target' in combined_train[
            'gc'].columns else None
        assert labels is not None, "Labels should be present in the GC data at this point!"

        if 'target' in combined_train['gc'].columns:
            combined_train['gc'].drop(columns=['target'], inplace=True)
        if 'target' in combined_train['go'].columns:
            combined_train['go'].drop(columns=['target'], inplace=True)

        num_train_positives = sum(list(labels.values()))
        num_train_negatives = len(labels) - num_train_positives

        num_pfam_pos = sum(combined_test_data['pfam']['gc']['target'])
        num_pfam_neg = len(combined_test_data['pfam']['gc']) - num_pfam_pos

        num_rcnt_pos = sum(combined_test_data['rcnt']['gc']['target'])
        num_rcnt_neg = len(combined_test_data['rcnt']['gc']) - num_rcnt_pos

        num_pharos_pos = sum(combined_test_data['pharos']['gc']['target'])
        num_pharos_neg = len(combined_test_data['pharos']['gc']) - num_pharos_pos

        data = [
            ["Training Data", num_train_positives, num_train_negatives, "-"],
            ["Pfam Test Data", num_pfam_pos, "-", num_pfam_neg],
            ["Recent Test Data", num_rcnt_pos, "-", num_rcnt_neg],
            ["Pharos Test Data", num_pharos_pos, "-", num_pharos_neg]
        ]

        # import pickle
        # # save the test_data to a pickle file
        # with open("../data/test_data/full_test_labels.pkl", 'wb') as f:
        #     pickle.dump(test_labels, f)

        # remove the column target from gc and go test data
        combined_test_data = self._clean_test_data(combined_test_data)

        headers = ["Data Source", "Approved Drug Targets", "Unlabelled Targets", "Putative Rejected Drug Targets"]

        print(tabulate(data, headers=headers, tablefmt="pretty"))

        return {
            "train": combined_train,
            "labels": labels,
            "genes": list(combined_genes),
            "num_features": combined_features,
            "config": config,
            "test_data": combined_test_data,
            "test_labels": test_labels,
            "test_labels_per_source": test_labels_per_source,
            "test_genes": combined_test_genes,
            "class_prior": float(class_prior)
        }

    def get_test_data(self, gc_data, go_data, pvc_data):
        """Extract test data from preprocessors and organize it properly."""
        if self.config['hyperparameters']['mode'] == 'eval':
            # Extract test target IDs for each source
            pfam_ids = gc_data.ensg_ids[gc_data.ensg_ids.isin(gc_data.drgbl_targets_pfam)]
            rcnt_ids = gc_data.ensg_ids[gc_data.ensg_ids.isin(gc_data.rcnt_targets_fda)]
            pharos_ids = gc_data.ensg_ids[gc_data.ensg_ids.isin(gc_data.chem_targets_pharos)]

            # Convert to lists for easier handling
            pfam_ensg = pfam_ids.tolist()
            rcnt_ensg = rcnt_ids.tolist()
            pharos_ensg = pharos_ids.tolist()

            # Extract positive test data
            pfam_pos_data_gc = gc_data.data[gc_data.data.index.isin(pfam_ensg)]
            pfam_pos_data_go = go_data.data[go_data.data.index.isin(pfam_ensg)]

            pvc_labels = pvc_data.data['labels']
            pfam_pos_data_pvc = {ensg: pvc_data.data[ensg] for ensg in pfam_ensg if ensg in list(pvc_labels.keys())}

            rcnt_pos_data_gc = gc_data.data[gc_data.data.index.isin(rcnt_ensg)]
            rcnt_pos_data_go = go_data.data[go_data.data.index.isin(rcnt_ensg)]
            rcnt_pos_data_pvc = {ensg: pvc_data.data[ensg] for ensg in rcnt_ensg if ensg in list(pvc_labels.keys())}

            pharos_pos_data_gc = gc_data.data[gc_data.data.index.isin(pharos_ensg)]
            pharos_pos_data_go = go_data.data[go_data.data.index.isin(pharos_ensg)]
            pharos_pos_data_pvc = {ensg: pvc_data.data[ensg] for ensg in pharos_ensg if ensg in list(pvc_labels.keys())}
            pharos_pos_data_gc.loc[:, 'target'] = 1
            pharos_pos_data_go['target'] = 1

            # Calculate class ratio
            num_pos = len(gc_data.data[gc_data.data['target'] == 1])
            num_neg = len(gc_data.data[gc_data.data['target'] == 0])
            class_prior = num_pos / (num_pos + num_neg)

            # Calculate needed negative samples for each test source
            num_pfam_neg = int(len(pfam_pos_data_gc) / class_prior)
            num_rcnt_neg = int(len(rcnt_pos_data_gc) / class_prior)
            num_pharos_neg = int(len(pharos_pos_data_gc) / class_prior)

            # Select and distribute negative samples
            total_negs = num_pfam_neg + num_rcnt_neg + num_pharos_neg

            negative_candidates = gc_data.data[gc_data.data['target'] == 0]
            negative_candidates = negative_candidates[negative_candidates['geneticConstraint'] < -0.90]

            if total_negs >= len(negative_candidates):
                negative_test_balance = negative_candidates
            else:
                negative_test_balance = negative_candidates.sample(n=total_negs,
                                                                   random_state=self.config['hyperparameters']['seed'])

            negative_test_ids = gc_data.ensg_ids[gc_data.ensg_ids.isin(negative_test_balance.index)]

            pfam_neg_ratio = float(num_pfam_neg / total_negs)
            rcnt_neg_ratio = float(num_rcnt_neg / total_negs)
            pharos_neg_ratio = float(num_pharos_neg / total_negs)

            num_pfam_neg = int(len(negative_test_ids) * pfam_neg_ratio)
            num_rcnt_neg = int(len(negative_test_ids) * rcnt_neg_ratio)
            num_pharos_neg = int(len(negative_test_ids) * pharos_neg_ratio)

            # Sample negative examples
            negative_test_ids = gc_data.ensg_ids[gc_data.ensg_ids.isin(negative_candidates.index)]
            negative_test_ids = negative_test_ids.to_frame()
            negative_test_ids.set_index('targetId', inplace=True)

            # Allocate negatives to each test source
            pfam_negs = negative_test_ids.sample(n=num_pfam_neg, random_state=self.config['hyperparameters']['seed'])
            negative_test_ids = negative_test_ids.drop(pfam_negs.index)
            pfam_neg_data_gc = gc_data.data[gc_data.data.index.isin(pfam_negs.index)]
            pfam_neg_data_go = go_data.data[go_data.data.index.isin(pfam_negs.index)]
            pfam_neg_data_pvc = {ensg: pvc_data.data[ensg] for ensg in pfam_negs.index if
                                 ensg in list(pvc_labels.keys())}

            rcnt_negs = negative_test_ids.sample(n=num_rcnt_neg, random_state=self.config['hyperparameters']['seed'])
            negative_test_ids = negative_test_ids.drop(rcnt_negs.index)
            rcnt_neg_data_gc = gc_data.data[gc_data.data.index.isin(rcnt_negs.index)]
            rcnt_neg_data_go = go_data.data[go_data.data.index.isin(rcnt_negs.index)]
            rcnt_neg_data_pvc = {ensg: pvc_data.data[ensg] for ensg in rcnt_negs.index if
                                 ensg in list(pvc_labels.keys())}

            pharos_negs = negative_test_ids.sample(n=num_pharos_neg,
                                                   random_state=self.config['hyperparameters']['seed'])
            pharos_neg_data_gc = gc_data.data[gc_data.data.index.isin(pharos_negs.index)]
            pharos_neg_data_go = go_data.data[go_data.data.index.isin(pharos_negs.index)]
            pharos_neg_data_pvc = {ensg: pvc_data.data[ensg] for ensg in pharos_negs.index if
                                   ensg in list(pvc_labels.keys())}
            pharos_neg_data_gc.loc[:, 'target'] = 0
            pharos_neg_data_go.loc[:, 'target'] = 0

            # Combine positive and negative data for each source
            pfam_data_gc = pd.concat([pfam_pos_data_gc, pfam_neg_data_gc])
            pfam_data_go = pd.concat([pfam_pos_data_go, pfam_neg_data_go])
            pfam_data_pvc = {**pfam_pos_data_pvc, **pfam_neg_data_pvc}

            rcnt_data_gc = pd.concat([rcnt_pos_data_gc, rcnt_neg_data_gc])
            rcnt_data_go = pd.concat([rcnt_pos_data_go, rcnt_neg_data_go])
            rcnt_data_pvc = {**rcnt_pos_data_pvc, **rcnt_neg_data_pvc}

            pharos_data_gc = pd.concat([pharos_pos_data_gc, pharos_neg_data_gc])
            pharos_data_go = pd.concat([pharos_pos_data_go, pharos_neg_data_go])
            pharos_data_pvc = {**pharos_pos_data_pvc, **pharos_neg_data_pvc}

            test_labels = {}

            all_pos_genes = set(pfam_ensg + rcnt_ensg + pharos_ensg)
            for gene in all_pos_genes:
                test_labels[gene] = 1

            all_neg_genes = set(list(pfam_negs.index) + list(rcnt_negs.index) + list(pharos_negs.index))
            for gene in all_neg_genes:
                test_labels[gene] = 0

            # Set up test data structure
            test_data = {
                "pfam": {
                    "gc": pfam_data_gc,
                    "go": pfam_data_go,
                    "pvc": pfam_data_pvc
                },
                "rcnt": {
                    "gc": rcnt_data_gc,
                    "go": rcnt_data_go,
                    "pvc": rcnt_data_pvc
                },
                "pharos": {
                    "gc": pharos_data_gc,
                    "go": pharos_data_go,
                    "pvc": pharos_data_pvc
                }
            }

            pfam_ids_all = pfam_data_gc.index.tolist()
            rcnt_ids_all = rcnt_data_gc.index.tolist()
            pharos_ids_all = pharos_data_gc.index.tolist()
            all_test_ids = pfam_ids_all + rcnt_ids_all + pharos_ids_all

            return {
                "test_data": test_data,
                "test_genes": {"pfam": pfam_ids_all, "rcnt": rcnt_ids_all, "pharos": pharos_ids_all},
                "all_test_ids": all_test_ids,
                "test_labels": test_labels,
                "class_prior": class_prior
            }


        elif self.config['hyperparameters']['mode'] == 'inference':
            seed = self.config['hyperparameters']['seed']

            # Get all gene IDs per modality
            gc_genes = set(gc_data.data.index.tolist())
            go_genes = set(go_data.data.index.tolist())
            pvc_genes = set(pvc_data.data['labels'].keys())

            # Intersection of all genes across the three modalities
            common_genes = gc_genes & go_genes & pvc_genes

            # Within this common set, identify unlabeled genes
            gc_unlabeled = set(gc_data.data.loc[list(common_genes)][gc_data.data['target'] == 0].index.tolist())
            go_unlabeled = set(
                go_data.data[
                    go_data.data.index.isin(common_genes) &
                    go_data.data.index.map(gc_data.labels) == 0
                    ].index.tolist()
            )
            pvc_unlabeled = set([ensg for ensg in common_genes if pvc_data.data['labels'][ensg] == 0])

            # Final set of unlabeled genes that are in all three modalities
            unlabeled_common = list(gc_unlabeled & go_unlabeled & pvc_unlabeled)

            random.seed(seed)
            random.shuffle(unlabeled_common)
            split_idx = int(len(unlabeled_common) * 0.8)
            train_ids = unlabeled_common[:split_idx]
            test_ids = unlabeled_common[split_idx:]

            # Extract train/test data aligned on common gene set
            gc_train = gc_data.data.loc[train_ids]
            gc_test = gc_data.data.loc[test_ids]
            go_train = go_data.data.loc[train_ids]
            go_test = go_data.data.loc[test_ids]
            pvc_train = {ensg: pvc_data.data[ensg] for ensg in train_ids}
            pvc_test = {ensg: pvc_data.data[ensg] for ensg in test_ids}

            test_labels = {gene: gc_data.labels[gene] for gene in common_genes}
            num_pos = sum(1 for gene in common_genes if gc_data.labels[gene] == 1)
            num_neg = sum(1 for gene in common_genes if gc_data.labels[gene] == 0)
            class_prior = num_pos / (num_pos + num_neg) if (num_pos + num_neg) > 0 else 0

            return {
                "test_data": {
                    "gc": gc_test,
                    "go": go_test,
                    "pvc": pvc_test
                },
                "train_data": {
                    "gc": gc_train,
                    "go": go_train,
                    "pvc": pvc_train
                },
                "all_test_ids": test_ids,
                "test_labels": test_labels,
                "class_prior": class_prior

            }
        else:
            raise ValueError(
                f"Invalid mode '{self.config['hyperparameters']['mode']}'. "
                "Expected 'eval' or 'inference'."
            )


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
    def __init__(self, data, labels, gene_names, dtype, variant_data=None, max_variants=None, test_source=False):
        self.data = data
        self.labels = labels
        self.gene_names = gene_names
        self.variant_data = variant_data
        self.max_variants = max_variants
        self.test_source = test_source
        self.torch_dtype = dtype

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
        self.torch_dtype = datasets['gc'].torch_dtype
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
                            if items[0].dtype in (torch.int64, torch.int32):
                                items = [item.to(torch.float32) for item in items]
                            else:
                                items = [item.to(self.torch_dtype) for item in items]
                            batch[modality][key] = torch.stack(items)
                        elif isinstance(items[0], (int, float, bool)):
                            batch[modality][key] = torch.tensor(items, dtype=self.torch_dtype)
                        else:
                            batch[modality][key] = items
                else:
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
