"""Multi-modal data pipeline for gene characterisation, ontology, and variant features."""
import random

import yaml
import pandas as pd

from varformer.data.features.gc import GeneCharacterisationPreprocessor
from varformer.data.features.go import GeneOntologyPreprocessor
from varformer.data.features.variants import PopulationVariantPreprocessor
from tabulate import tabulate


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
        gcp = GeneCharacterisationPreprocessor(config=self.config)
        print("Gene characterisation features preprocessed!\n")
        return gcp

    def open_go_data(self, gc_data):
        gop = GeneOntologyPreprocessor(config=self.config, gcp=gc_data)
        print("Gene ontology features preprocessed!\n")
        return gop

    def open_pvc_data(self, gc_data):
        pvc = PopulationVariantPreprocessor(config=self.config, gcp=gc_data)
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
        ensg_gc = set(gc_data.data.index.tolist())
        ensg_go = set(go_data.data.index.tolist())

        if pvc_data is not None:
            ensg_pvc = set(pvc_data.data.keys())
            if 'labels' in ensg_pvc:
                ensg_pvc.remove('labels')
            common_genes = ensg_gc.intersection(ensg_go).intersection(ensg_pvc)
            print(f"Original gene counts: GC={len(ensg_gc)}, GO={len(ensg_go)}, PVC={len(ensg_pvc)}")
        else:
            ensg_pvc = None
            common_genes = ensg_gc.intersection(ensg_go)
            print(f"Original gene counts: GC={len(ensg_gc)}, GO={len(ensg_go)}, PVC=None")

        print(f"Common genes across active modalities: {len(common_genes)}")

        # Keep only common genes in each modality
        data['gc'].data = data['gc'].data[data['gc'].data.index.isin(common_genes)]
        data['go'].data = data['go'].data[data['go'].data.index.isin(common_genes)]

        data['go'].data = data['go'].data.loc[:, (data['go'].data != 0).any(axis=0)]

        # For PVC data which is in dictionary format
        if pvc_data is not None:
            for gene in list(pvc_data.data.keys()):
                if gene != 'labels' and gene not in common_genes:
                    pvc_data.data.pop(gene)
                    if gene in pvc_data.labels:
                        pvc_data.labels.pop(gene)

        # Verify all active modalities now have the same number of genes
        gc_count = len(data['gc'].data)
        go_count = len(data['go'].data)

        if pvc_data is not None:
            pvc_count = len(pvc_data.data) - (1 if 'labels' in pvc_data.data else 0)
            assert gc_count == go_count == pvc_count, "Gene counts don't match across modalities"
        else:
            assert gc_count == go_count, "Gene counts don't match across modalities"

        gc_data = data['gc']
        go_data = data['go']
        pvc_data = data['pvc']

        test_data_result = self.get_test_data(gc_data, go_data, pvc_data)

        if self.config['hyperparameters']['mode'] == 'inference':
            return test_data_result

        elif self.config['hyperparameters']['mode'] == 'eval':
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
                if preprocessor is None:
                    continue
                feature_data = preprocessor.data
                combined_features += preprocessor.num_features
                if isinstance(feature_data, pd.DataFrame):
                    feature_data = feature_data[~feature_data.index.isin(combined_test_genes)]
                    config = preprocessor.config
                elif isinstance(feature_data, dict):
                    feature_data = {gene: feature_data[gene] for gene in feature_data if
                                    gene not in combined_test_genes}
                    feature_data.pop('labels', None)
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

            labels = dict(zip(combined_train['gc'].index, combined_train['gc']['target'])) if 'target' in \
                                                                                              combined_train[
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

        else:
            raise ValueError(
                f"Invalid mode '{self.config['hyperparameters']['mode']}'. "
                "Expected 'eval' or 'inference'."
            )

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

            if pvc_data is not None:
                pvc_labels = pvc_data.data['labels']
                pfam_pos_data_pvc = {ensg: pvc_data.data[ensg] for ensg in pfam_ensg if ensg in list(pvc_labels.keys())}
                rcnt_pos_data_pvc = {ensg: pvc_data.data[ensg] for ensg in rcnt_ensg if ensg in list(pvc_labels.keys())}
                pharos_pos_data_pvc = {ensg: pvc_data.data[ensg] for ensg in pharos_ensg if
                                       ensg in list(pvc_labels.keys())}
            else:
                pvc_labels = None
                pfam_pos_data_pvc = None
                rcnt_pos_data_pvc = None
                pharos_pos_data_pvc = None

            rcnt_pos_data_gc = gc_data.data[gc_data.data.index.isin(rcnt_ensg)]
            rcnt_pos_data_go = go_data.data[go_data.data.index.isin(rcnt_ensg)]

            pharos_pos_data_gc = gc_data.data[gc_data.data.index.isin(pharos_ensg)]
            pharos_pos_data_go = go_data.data[go_data.data.index.isin(pharos_ensg)]
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
            if pvc_data is not None:
                pfam_neg_data_pvc = {ensg: pvc_data.data[ensg] for ensg in pfam_negs.index if
                                     ensg in list(pvc_labels.keys())}
            else:
                pfam_neg_data_pvc = None

            rcnt_negs = negative_test_ids.sample(n=num_rcnt_neg, random_state=self.config['hyperparameters']['seed'])
            negative_test_ids = negative_test_ids.drop(rcnt_negs.index)
            rcnt_neg_data_gc = gc_data.data[gc_data.data.index.isin(rcnt_negs.index)]
            rcnt_neg_data_go = go_data.data[go_data.data.index.isin(rcnt_negs.index)]
            if pvc_data is not None:
                rcnt_neg_data_pvc = {ensg: pvc_data.data[ensg] for ensg in rcnt_negs.index if
                                     ensg in list(pvc_labels.keys())}
            else:
                rcnt_neg_data_pvc = None

            pharos_negs = negative_test_ids.sample(n=num_pharos_neg,
                                                   random_state=self.config['hyperparameters']['seed'])
            pharos_neg_data_gc = gc_data.data[gc_data.data.index.isin(pharos_negs.index)]
            pharos_neg_data_go = go_data.data[go_data.data.index.isin(pharos_negs.index)]
            if pvc_data is not None:
                pharos_neg_data_pvc = {ensg: pvc_data.data[ensg] for ensg in pharos_negs.index if
                                       ensg in list(pvc_labels.keys())}
            else:
                pharos_neg_data_pvc = None
            pharos_neg_data_gc.loc[:, 'target'] = 0
            pharos_neg_data_go.loc[:, 'target'] = 0

            # Combine positive and negative data for each source
            pfam_data_gc = pd.concat([pfam_pos_data_gc, pfam_neg_data_gc])
            pfam_data_go = pd.concat([pfam_pos_data_go, pfam_neg_data_go])

            rcnt_data_gc = pd.concat([rcnt_pos_data_gc, rcnt_neg_data_gc])
            rcnt_data_go = pd.concat([rcnt_pos_data_go, rcnt_neg_data_go])

            pharos_data_gc = pd.concat([pharos_pos_data_gc, pharos_neg_data_gc])
            pharos_data_go = pd.concat([pharos_pos_data_go, pharos_neg_data_go])

            if pvc_data is not None:
                pfam_data_pvc = {**pfam_pos_data_pvc, **pfam_neg_data_pvc}
                rcnt_data_pvc = {**rcnt_pos_data_pvc, **rcnt_neg_data_pvc}
                pharos_data_pvc = {**pharos_pos_data_pvc, **pharos_neg_data_pvc}
            else:
                pfam_data_pvc = None
                rcnt_data_pvc = None
                pharos_data_pvc = None

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
            random.seed(seed)
            enforced_split_size = 822  # Split size equivalent to eval mode

            # Get all gene IDs per modality
            gc_genes = set(gc_data.data.index.tolist())
            go_genes = set(go_data.data.index.tolist())
            pvc_genes = set(pvc_data.data['labels'].keys())

            # Intersection of all genes across the three modalities
            common_genes = list(gc_genes & go_genes & pvc_genes)

            # Get labeled and unlabeled genes
            labeled_genes = []
            unlabeled_genes = []

            for gene in common_genes:
                if gc_data.labels[gene] == 1:
                    labeled_genes.append(gene)
                else:
                    unlabeled_genes.append(gene)

            # Shuffle unlabeled genes for random distribution across splits
            random.shuffle(unlabeled_genes)

            # Calculate number of splits based on enforced split size for unlabeled genes
            num_splits = len(unlabeled_genes) // enforced_split_size
            if len(unlabeled_genes) % enforced_split_size != 0:
                num_splits += 1

            splits = []

            # Generate multiple non-overlapping splits for unlabeled genes
            for i in range(num_splits):
                start_idx = i * enforced_split_size
                end_idx = min((i + 1) * enforced_split_size, len(unlabeled_genes))

                # Get current split's unlabeled test genes
                current_unlabeled_test = unlabeled_genes[start_idx:end_idx]

                # Include all labeled genes and remaining unlabeled genes in training
                all_genes = common_genes.copy()
                test_genes = current_unlabeled_test

                # Training set: All genes except current unlabeled test genes
                train_genes = [gene for gene in all_genes if gene not in current_unlabeled_test]

                # Extract data for current split
                gc_train = gc_data.data.loc[train_genes]
                gc_test = gc_data.data.loc[test_genes]
                go_train = go_data.data.loc[train_genes]
                go_test = go_data.data.loc[test_genes]

                pvc_train = {ensg: pvc_data.data[ensg] for ensg in train_genes}
                pvc_test = {ensg: pvc_data.data[ensg] for ensg in test_genes}

                # Labels for this split
                train_labels = {gene: gc_data.labels[gene] for gene in train_genes}
                test_labels = {gene: gc_data.labels[gene] for gene in test_genes}

                # Calculate class prior
                num_pos = sum(1 for gene in common_genes if gc_data.labels[gene] == 1)
                num_neg = sum(1 for gene in common_genes if gc_data.labels[gene] == 0)
                class_prior = num_pos / (num_pos + num_neg) if (num_pos + num_neg) > 0 else 0

                # Create a dictionary of all genes and their labels
                all_labels = {gene: gc_data.labels[gene] for gene in common_genes}

                # Add labels to PVC data
                pvc_train_with_labels = pvc_train.copy()
                pvc_train_with_labels['labels'] = {gene: train_labels[gene] for gene in train_genes}
                pvc_test_with_labels = pvc_test.copy()
                pvc_test_with_labels['labels'] = {gene: test_labels[gene] for gene in test_genes}

                # Append this split to the list
                splits.append({
                    "test_data": {
                        "gc": gc_test,
                        "go": go_test,
                        "pvc": pvc_test_with_labels
                    },
                    "train": {
                        "gc": gc_train,
                        "go": go_train,
                        "pvc": pvc_train_with_labels
                    },
                    "test_labels": test_labels,
                    "test_genes": test_genes,
                    "train_genes": train_genes,
                    "labels": all_labels,
                    "class_prior": class_prior,
                    "config": gc_data.config
                })

            return splits
        else:
            raise ValueError(
                f"Invalid mode '{self.config['hyperparameters']['mode']}'. "
                "Expected 'eval' or 'inference'."
            )
