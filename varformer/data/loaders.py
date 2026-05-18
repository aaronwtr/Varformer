"""ModelPreprocessorEval and ModelPreprocessorInference for data loading and model initialisation."""
import pickle as pkl
import types

import torch
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# Legacy `dl.<name>` references resolve via this synthetic namespace pointing at the
# new package locations, instead of importing the src/dataloader shim. This lets the
# SDK run without src/ on sys.path.
from varformer.data import datasets as _datasets
from varformer.data import samplers as _samplers

dl = types.SimpleNamespace(
    MultiModalData=_datasets.MultiModalData,
    MultiModalDataLoader=_samplers.MultiModalDataLoader,
    VarformerDataset=_datasets.VarformerDataset,
    DrugTargetData=_datasets.DrugTargetData,
    SynchronizedMultiModalBatchSampler=_samplers.SynchronizedMultiModalBatchSampler,
)

from varformer.training.lightning_module import VarformerLightningModule as MultiModalLightningTargetIdentifier


# Model preprocessing
class ModelPreprocessorEval:
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.gc_data = data['train']['gc']
        self.go_data = data['train']['go']
        self.pvc_data = data['train'].get('pvc', None)
        if self.pvc_data is not None:
            self.pvc_data = self.pvc_data.copy()
        self.labels = data["labels"]
        self.test_labels = data["test_labels"]
        self.test_labels_per_source = data["test_labels_per_source"]
        self.genes = data['genes']
        self.num_features = data['num_features']
        self.test_data = data['test_data']
        self.test_genes = data['test_genes']
        self.train_genes, self.val_genes = train_test_split(self.genes, test_size=0.2,
                                                            random_state=config['hyperparameters']['seed'])
        self.class_prior = data['class_prior']
        self.torch_dtype = torch.bfloat16 if config['hyperparameters']['precision'] == 'bf16-mixed' else torch.float32
        torch.set_default_dtype(self.torch_dtype)

    def model_init(self):
        gc_train_raw = self.gc_data.loc[self.train_genes, :]
        gc_val_raw = self.gc_data.loc[self.val_genes, :]

        go_train_raw = self.go_data.loc[self.train_genes, :]
        go_val_raw = self.go_data.loc[self.val_genes, :]

        train_raw = {
            'gc': gc_train_raw,
            'go': go_train_raw,
        }

        val_raw = {
            'gc': gc_val_raw,
            'go': go_val_raw,
        }

        if self.pvc_data is not None:
            pvc_train_raw = {k: v for k, v in self.pvc_data.items() if k in self.train_genes}
            pvc_val_raw = {k: v for k, v in self.pvc_data.items() if k in self.val_genes}
            train_raw['pvc'] = pvc_train_raw
            val_raw['pvc'] = pvc_val_raw

        model, train_combined, val_combined, test_combined, hyperparameters, accelerator = self.initialise_model(
            train_raw,
            val_raw,
            self.labels,
            self.test_labels,
            self.train_genes,
            self.val_genes,
            self.test_labels_per_source,
            self.test_data,
            self.torch_dtype,
            self.config
        )

        return model, train_combined, val_combined, test_combined, hyperparameters, accelerator

    def initialise_model(self, train_raw, val_raw, labels, test_labels, train_genes, val_genes, test_genes, test,
                         torch_dtype, config):
        hyperparams = config['hyperparameters']
        (train_combined, val_combined, test_combined,
         num_samples_per_class) = self.normalise_data(train_raw, val_raw, labels, test_labels, train_genes, val_genes,
                                                      test_genes, test, torch_dtype, config)

        use_pvc = 'pvc' in train_raw

        if use_pvc:
            max_genes_pvc = max([train_raw['pvc'][gene].shape[0] for gene in train_raw['pvc'].keys()])
            with open(config['paths']['MISSENSE_MAP'], "rb") as f:
                missense_map = pkl.load(f)
            num_mutations = len(missense_map)
        else:
            max_genes_pvc = 0
            num_mutations = 0

        gc_features_dim = train_raw['gc'].shape[1]
        go_features_dim = train_raw['go'].shape[1]

        model = MultiModalLightningTargetIdentifier(
            config=config,
            num_features_gc=gc_features_dim,
            num_features_go=go_features_dim,
            num_mutations=num_mutations,
            max_seq_len=hyperparams['max_seq_len'],
            num_genes=max_genes_pvc,
            num_samples_per_class=num_samples_per_class,
            class_prior=self.class_prior,
            use_pvc=use_pvc
        )

        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

        hyperparameters = dict(
            depth=hyperparams['depth_cls_head'],
            lr=hyperparams['lr_start'],
            batch_size=hyperparams['batch_size'],
            optimizer=hyperparams['optimizer'],
            epochs=hyperparams['epochs'],
            dropout=hyperparams['dropout'],
            gc_width=hyperparams['gc_width'],
            go_width=hyperparams['go_width'],
            weight_decay=hyperparams['weight_decay']
        )

        return model, train_combined, val_combined, test_combined, hyperparameters, accelerator

    @staticmethod
    def normalise_data(train_raw, val_raw, labels, test_labels, train_genes, val_genes, test_genes, test_raw,
                       torch_dtype, config):
        hparams = config['hyperparameters']

        train_datasets = {}
        val_datasets = {}
        test_datasets = {key: {} for key in test_raw.keys()}
        scalers = {}

        for module_str, train_data in train_raw.items():
            if module_str != "pvc":
                val_norm = val_raw[module_str].values
                train_norm = train_data.values

                # scaler = MinMaxScaler()
                # train_norm = scaler.fit_transform(train_norm)
                # val_norm = scaler.transform(val_norm)
                # scalers[module_str] = scaler

                train_norm = {gene: train_norm[i] for i, gene in enumerate(train_genes)}
                val_norm = {gene: val_norm[i] for i, gene in enumerate(val_genes)}

                train_datasets[module_str] = dl.MultiModalData(
                    data=train_norm,
                    labels=labels,
                    gene_names=train_genes,
                    dtype=torch_dtype
                )

                val_datasets[module_str] = dl.MultiModalData(
                    data=val_norm,
                    labels=labels,
                    gene_names=val_genes,
                    dtype=torch_dtype
                )

                for key, modalities in test_raw.items():
                    # normed = scaler.transform(modalities[module_str].values)
                    # normed = {gene: normed[i] for i, gene in enumerate(test_genes[key][module_str])}
                    test_data = {gene: modalities[module_str].values[i] for i, gene in enumerate(test_genes[key])}
                    test_datasets[key][module_str] = dl.MultiModalData(
                        data=test_data,
                        labels=test_labels,
                        gene_names=test_genes[key],
                        dtype=torch_dtype,
                        test_source=key
                    )
            else:
                train_datasets[module_str] = dl.MultiModalData(
                    data=None,
                    labels=None,
                    gene_names=train_genes,
                    dtype=torch_dtype,
                    variant_data={'data': train_data, 'labels': labels},
                    max_variants=hparams['max_seq_len']
                )

                val_datasets[module_str] = dl.MultiModalData(
                    data=None,
                    labels=None,
                    gene_names=val_genes,
                    dtype=torch_dtype,
                    variant_data={'data': val_raw[module_str], 'labels': labels},
                    max_variants=hparams['max_seq_len']
                )

                for key, modalities in test_raw.items():
                    test_datasets[key][module_str] = dl.MultiModalData(
                        data=None,
                        labels=None,
                        gene_names=test_genes[key],
                        dtype=torch_dtype,
                        variant_data={
                            'data': modalities[module_str],
                            'labels': test_labels,
                            'test_source': key
                        },
                        max_variants=hparams['max_seq_len'],
                        test_source=key
                    )

        #get from the test_datasets['pharos']['gc'] the gene names and the labels
        test_gene_names = test_datasets['pharos']['gc'].gene_names
        test_labels = test_datasets['pharos']['gc'].labels
        subset_labels_test = {gene: test_labels[gene] for gene in test_gene_names if gene in test_labels}
        subset_labels = {gene: labels[gene] for gene in test_gene_names if gene in labels}

        train_loader = dl.MultiModalDataLoader(
            datasets=train_datasets,
            batch_size=hparams['batch_size'],
            shuffle=True
        )

        val_loader = dl.MultiModalDataLoader(
            datasets=val_datasets,
            batch_size=hparams['batch_size'],
            shuffle=False
        )

        test_loaders = {}
        for key in test_raw.keys():
            if len(next(iter(test_datasets[key].values()))) <= 1000:
                test_loaders[key] = dl.MultiModalDataLoader(
                    datasets=test_datasets[key],
                    batch_size=len(next(iter(test_datasets[key].values()))),
                    shuffle=False
                )
            else:
                test_loaders[key] = dl.MultiModalDataLoader(
                    datasets=test_datasets[key],
                    batch_size=hparams['batch_size'],
                    shuffle=False
                )

        num_maj_samples, num_min_samples = next(iter(train_datasets.values())).samples_per_class()

        return train_loader, val_loader, test_loaders, (num_maj_samples, num_min_samples)


class ModelPreprocessorInference:
    def __init__(self, config, consolidated_data, pvc_data, gene_names):
        self.config = config
        self.data_raw = consolidated_data
        self.pvc_data = pvc_data
        self.gene_names = gene_names
        self.torch_dtype = torch.bfloat16 if config['hyperparameters']['precision'] == 'bf16-mixed' else torch.float32
        torch.set_default_dtype(self.torch_dtype)

    def model_init(self):
        # Create the unlabeled loader
        unlabeled_loader, num_samples = self.create_unlabeled_loader(
            self.data_raw,
            self.pvc_data,
            self.gene_names,
            self.torch_dtype,
            self.config
        )

        test_loaders = self.create_test_loaders(
            self.data_raw,
            self.pvc_data,
            self.torch_dtype,
            self.config
        )

        # Initialize model dimensions
        gc_features_dim = self.data_raw['gc'].shape[1] - 1 if 'target' in self.data_raw['gc'].columns else self.data_raw['gc'].shape[1]
        go_features_dim = self.data_raw['go'].shape[1] - 1 if 'target' in self.data_raw['go'].columns else self.data_raw['go'].shape[1]

        with open(self.config['paths']['MISSENSE_MAP'], 'rb') as f:
            missense_map = pkl.load(f)
        num_mutations = len(missense_map)

        model = MultiModalLightningTargetIdentifier(
            config=self.config,
            num_features_gc=gc_features_dim,
            num_features_go=go_features_dim,
            num_mutations=num_mutations,
            max_seq_len=self.config['hyperparameters']['max_seq_len'],
            num_genes=len(self.gene_names),
            num_samples_per_class=(len(self.gene_names), 0),  # all unlabeled = class 0
            class_prior={0: 1.0, 1: 0.0}
        )

        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
        return unlabeled_loader, test_loaders, gc_features_dim, go_features_dim, num_genes, num_mutations

    def initialise_model(self, train_raw, val_raw, labels, test_labels, train_genes, val_genes, test_genes_dict,
                         test_data_dict, torch_dtype, config):
        hyperparams = config['hyperparameters']
        unlabeled_loader, num_samples_per_class = ModelPreprocessorInference.create_unlabeled_loader(
            train_raw, val_raw, labels, test_labels, train_genes, val_genes,
            test_genes_dict, test_data_dict, torch_dtype, config
        )

        # Determine max_genes_pvc (max variants per gene in training set for PVC)
        # Ensure pvc data in train_raw is not empty and genes have data
        max_genes_pvc = 0
        if 'pvc' in train_raw and train_raw['pvc']:
            non_empty_pvc_genes = [gene for gene in train_raw['pvc'] if train_raw['pvc'][gene].nelement() > 0]
            if non_empty_pvc_genes:
                max_genes_pvc = max([train_raw['pvc'][gene].shape[0] for gene in non_empty_pvc_genes])

        with open(config['paths']['MISSENSE_MAP'], "rb") as f:
            missense_map = pkl.load(f)
        num_mutations = len(missense_map)

        if 'target' in train_raw['gc'].columns:
            gc_features_dim = train_raw['gc'].shape[1] - 1  # -1 for target column if present before normalise
        else:
            gc_features_dim = train_raw['gc'].shape[1]

        if 'target' in train_raw['go'].columns:
            go_features_dim = train_raw['go'].shape[1] - 1
        else:
            go_features_dim = train_raw['go'].shape[1]

        model = MultiModalLightningTargetIdentifier(
            config=config,
            num_features_gc=gc_features_dim,
            num_features_go=go_features_dim,
            num_mutations=num_mutations,
            max_seq_len=hyperparams['max_seq_len'],
            num_genes=max_genes_pvc,  # This might need to be num_unique_gene_ids for embedding
            num_samples_per_class=num_samples_per_class,
            class_prior=self.class_prior  # Use actual class_prior from data_split
        )

        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

        hyperparameters_log = dict(
            depth=hyperparams['depth_cls_head'],
            lr=hyperparams['lr_start'],
            batch_size=hyperparams['batch_size'],
            optimizer=hyperparams['optimizer'],
            epochs=hyperparams['epochs'],
            dropout=hyperparams['dropout'],
            gc_width=hyperparams['gc_width'],
            go_width=hyperparams['go_width'],
            weight_decay=hyperparams['weight_decay']
        )

        return model, unlabeled_loader, hyperparameters_log, accelerator

    @staticmethod
    def create_unlabeled_loader(consolidated_data, pvc_data, gene_names, torch_dtype, config):
        hparams = config['hyperparameters']

        unlabeled_datasets = {}

        # gc and go modalities
        for modality in ['gc', 'go']:
            data_dict = {
                gene: consolidated_data[modality].loc[gene].drop(labels=['target'], errors='ignore').values.flatten()
                for gene in gene_names
            }

            unlabeled_datasets[modality] = dl.MultiModalData(
                data=data_dict,
                labels={gene: 0 for gene in gene_names},
                gene_names=gene_names,
                dtype=torch_dtype
            )

        # pvc modality
        unlabeled_datasets['pvc'] = dl.MultiModalData(
            data=None,
            labels=None,
            gene_names=gene_names,
            dtype=torch_dtype,
            variant_data={'data': {gene: pvc_data[gene] for gene in gene_names},
                          'labels': {gene: 0 for gene in gene_names}},
            max_variants=hparams['max_seq_len']
        )

        dataset_size = len(gene_names)

        unlabeled_loader = dl.MultiModalDataLoader(
            datasets=unlabeled_datasets,
            batch_size=config['hyperparameters']['batch_size'],  # all at once
            shuffle=False
        )

        return unlabeled_loader, dataset_size

    @staticmethod
    def create_test_loaders(config, consolidated_data, pvc_data, torch_dtype):
        """
        Create dataloaders for test genes (approved targets) from pfam, rcnt, and pharos test sets.

        Args:
            config: Configuration dictionary
            consolidated_data: Dictionary with 'gc', 'go' dataframes containing ALL genes (train + test)
            pvc_data: Dictionary mapping gene_id -> variant tensor
            torch_dtype: Torch data type string ('bf16-mixed' or 'float32')

        Returns:
            Dictionary of test loaders: {'pfam': loader, 'rcnt': loader, 'pharos': loader}
        """
        import pickle

        # Convert dtype string to torch dtype
        if torch_dtype == 'bf16-mixed':
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        test_loaders = {}

        # Load test gene IDs from pickle file
        test_labels_file = config['paths'].get('TEST_LABELS_FILE')

        try:
            with open(test_labels_file, 'rb') as f:
                test_gene_ids = pickle.load(f)
            print(f"  Loaded test gene IDs from {test_labels_file}")
        except FileNotFoundError:
            print(f"  Test labels file not found: {test_labels_file}")
            return {}

        # Create loader for each test set
        for test_name in ['pfam', 'rcnt', 'pharos']:
            if test_name not in test_gene_ids:
                print(f"  Skipping {test_name}: not in test labels file")
                continue

            test_genes = test_gene_ids[test_name]

            # Filter to genes available in all modalities
            available_genes = [
                gene for gene in test_genes
                if gene in consolidated_data['gc'].index and
                   gene in consolidated_data['go'].index and
                   gene in pvc_data
            ]

            if len(available_genes) == 0:
                print(f"  Skipping {test_name}: no genes available in data ({len(test_genes)} total)")
                continue

            print(f"  Creating {test_name} loader: {len(available_genes)}/{len(test_genes)} genes")

            # Create datasets (same pattern as unlabeled)
            gc_data_dict = {
                gene: consolidated_data['gc'].loc[gene].drop(labels=['target'], errors='ignore').values.flatten()
                for gene in available_genes
            }

            go_data_dict = {
                gene: consolidated_data['go'].loc[gene].drop(labels=['target'], errors='ignore').values.flatten()
                for gene in available_genes
            }

            # Labels are all 1 (positive/approved targets)
            labels = {gene: 1 for gene in available_genes}

            gc_dataset = dl.MultiModalData(
                data=gc_data_dict,
                labels=labels,
                gene_names=available_genes,
                dtype=dtype
            )

            go_dataset = dl.MultiModalData(
                data=go_data_dict,
                labels=labels,
                gene_names=available_genes,
                dtype=dtype
            )

            pvc_dataset = dl.MultiModalData(
                data=None,
                labels=None,
                gene_names=available_genes,
                dtype=dtype,
                variant_data={
                    'data': {gene: pvc_data[gene] for gene in available_genes},
                    'labels': labels
                },
                max_variants=config['hyperparameters']['max_seq_len']
            )

            # Create loader
            test_loaders[test_name] = dl.MultiModalDataLoader(
                datasets={'gc': gc_dataset, 'go': go_dataset, 'pvc': pvc_dataset},
                batch_size=min(32, len(available_genes)),
                shuffle=False
            )

        print(f"Created {len(test_loaders)} test loaders: {list(test_loaders.keys())}")
        return test_loaders
