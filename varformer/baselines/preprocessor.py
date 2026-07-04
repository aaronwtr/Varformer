"""Feature preprocessing for the logistic-regression baseline."""
import numpy as np

from sklearn.model_selection import train_test_split

from varformer.data.features.variants import extract_pvc_features


class LogisticRegressionPreprocessor:
    def __init__(self, config, data):
        """
        Initialize the preprocessor for logistic regression model

        Args:
            config: Configuration dictionary
            data: Data dictionary containing train, test data and other metadata
        """
        self.config = config
        self.data = data
        self.gc_data = data['train']['gc']
        self.go_data = data['train']['go']
        self.pvc_data = data['train'].get('pvc', None)
        if self.pvc_data is not None:
            self.pvc_data = self.pvc_data.copy()
        self.labels = self.data["labels"]
        self.test_labels = data["test_labels"]
        self.genes = data['genes']
        self.num_features = data['num_features']
        self.test_data = data['test_data']
        self.test_genes = data['test_genes']
        self.train_genes, self.val_genes = train_test_split(self.genes, test_size=0.2,
                                                            random_state=config['hyperparameters']['seed'])
        self.class_prior = data['class_prior']
        self.scalers = {}
        self.max_variants = config['hyperparameters'].get('max_seq_len', 100)

    def prepare_features(self):
        """
        Prepare features for logistic regression by processing and combining
        gene-centric (gc), gene ontology (go), and protein variant calls (pvc) data

        Returns:
            Dictionary containing processed train, validation, and test data
            with features and labels
        """
        print("Preparing features for logistic regression...")

        # Split data into train and validation sets
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

        # Process and combine the features
        processed_data = self.process_and_combine_features(
            train_raw,
            val_raw,
            self.labels,
            self.test_labels,
            self.train_genes,
            self.val_genes,
            self.test_genes,
            self.test_data
        )

        return processed_data

    def process_and_combine_features(self, train_raw, val_raw, labels, test_labels, train_genes, val_genes, test_genes,
                                     test_raw):
        """
        Process and combine features from different modalities for logistic regression

        Args:
            train_raw: Raw training data for different modalities
            val_raw: Raw validation data for different modalities
            labels: Training and validation labels
            test_labels: Test labels
            train_genes: List of genes in training set
            val_genes: List of genes in validation set
            test_genes: Dictionary of genes in test sets
            test_raw: Raw test data for different modalities

        Returns:
            Dictionary containing processed data for training, validation, and testing
        """
        # Initialize dictionaries to store processed data
        train_features = {}
        train_labels_list = []
        val_features = {}
        val_labels_list = []
        test_features = {test_set: {} for test_set in test_raw.keys()}
        test_labels_dict = {test_set: [] for test_set in test_raw.keys()}

        # Process gene-centric (gc) and gene ontology (go) features
        for module_str in ['gc', 'go']:
            # Scale the features
            # scaler = MinMaxScaler()
            # train_norm = scaler.fit_transform(train_raw[module_str].values)
            # val_norm = scaler.transform(val_raw[module_str].values)
            # self.scalers[module_str] = scaler
            train_feat = train_raw[module_str].values
            val_feat = val_raw[module_str].values

            train_features[module_str] = {train_genes[i]: train_feat[i] for i in range(len(train_genes))}
            val_features[module_str] = {val_genes[i]: val_feat[i] for i in range(len(val_genes))}

            # Process test data for each test set
            for test_set, modalities in test_raw.items():
                test_feat_df = modalities[module_str]
                test_feat = test_feat_df.values
                test_feat_genes = test_feat_df.index.tolist()
                test_features[test_set][module_str] = {test_feat_genes[i]: test_feat[i] for i in
                                                       range(len(test_feat_genes))}

        # Process protein variant calls (pvc) features using the dedicated function
        print("Processing PVC features...")
        train_features['pvc'] = self.process_pvc_batch(train_genes, train_raw['pvc'], self.max_variants)
        val_features['pvc'] = self.process_pvc_batch(val_genes, val_raw['pvc'], self.max_variants)

        for test_set, modalities in test_raw.items():
            test_genes_list = list(test_features[test_set]['gc'].keys())  # Using gc gene list for reference
            test_features[test_set]['pvc'] = self.process_pvc_batch(test_genes_list, modalities['pvc'],
                                                                     self.max_variants)

        # Combine features from all modalities into a single feature vector for each gene
        print("Combining features from all modalities...")
        combined_train_features = {}
        combined_val_features = {}
        combined_test_features = {test_set: {} for test_set in test_raw.keys()}

        # Create feature arrays ensuring all features are available
        for gene in train_genes:
            if gene in labels:
                gc_feat = train_features['gc'].get(gene, np.zeros(train_raw['gc'].shape[1]))
                go_feat = train_features['go'].get(gene, np.zeros(train_raw['go'].shape[1]))
                pvc_feat = train_features['pvc'].get(gene, np.zeros(10))  # 10 PVC features
                combined_train_features[gene] = np.concatenate([gc_feat, go_feat, pvc_feat])
                train_labels_list.append((gene, labels[gene]))

        for gene in val_genes:
            if gene in labels:
                gc_feat = val_features['gc'].get(gene, np.zeros(val_raw['gc'].shape[1]))
                go_feat = val_features['go'].get(gene, np.zeros(val_raw['go'].shape[1]))
                pvc_feat = val_features['pvc'].get(gene, np.zeros(10))
                combined_val_features[gene] = np.concatenate([gc_feat, go_feat, pvc_feat])
                val_labels_list.append((gene, labels[gene]))

        for test_set in test_raw.keys():
            test_gc_shape = test_raw[test_set]['gc'].shape[1]
            test_go_shape = test_raw[test_set]['go'].shape[1]

            for gene in list(test_features[test_set]['gc'].keys()):
                if gene in test_labels:
                    gc_feat = test_features[test_set]['gc'].get(gene, np.zeros(test_gc_shape))
                    go_feat = test_features[test_set]['go'].get(gene, np.zeros(test_go_shape))
                    pvc_feat = test_features[test_set]['pvc'].get(gene, np.zeros(10))
                    combined_test_features[test_set][gene] = np.concatenate([gc_feat, go_feat, pvc_feat])
                    test_labels_dict[test_set].append((gene, test_labels[gene]))

        # Convert to numpy arrays for scikit-learn
        X_train = np.array([combined_train_features[gene[0]] for gene in train_labels_list])
        y_train = np.array([label[1] for label in train_labels_list])

        X_val = np.array([combined_val_features[gene[0]] for gene in val_labels_list])
        y_val = np.array([label[1] for label in val_labels_list])

        X_test = {}
        y_test = {}
        test_gene_lists = {}
        for test_set in test_raw.keys():
            X_test[test_set] = np.array(
                [combined_test_features[test_set][gene[0]] for gene in test_labels_dict[test_set]])
            y_test[test_set] = np.array([label[1] for label in test_labels_dict[test_set]])
            test_gene_lists[test_set] = [gene[0] for gene in test_labels_dict[test_set]]

        # Handle missing values
        X_train = np.nan_to_num(X_train)
        X_val = np.nan_to_num(X_val)
        for test_set in X_test:
            X_test[test_set] = np.nan_to_num(X_test[test_set])

        # Log feature dimensions
        print(f"Features dimensions - Train: {X_train.shape}, Val: {X_val.shape}")
        for test_set in X_test:
            print(f"Test {test_set}: {X_test[test_set].shape}")

        # Document the feature set composition for interpretability
        feature_composition = {
            'gc_features': train_raw['gc'].shape[1],
            'go_features': train_raw['go'].shape[1],
            'pvc_features': 10,
            'total_features': X_train.shape[1]
        }
        print(f"Feature composition: {feature_composition}")

        return {
            'train': {'X': X_train, 'y': y_train, 'genes': [gene[0] for gene in train_labels_list]},
            'val': {'X': X_val, 'y': y_val, 'genes': [gene[0] for gene in val_labels_list]},
            'test': {test_set: {'X': X_test[test_set], 'y': y_test[test_set], 'genes': test_gene_lists[test_set]}
                     for test_set in test_raw.keys()},
            'feature_composition': feature_composition
        }

    @staticmethod
    def process_pvc_batch(genes, pvc_data, max_variants=100):
        """
        Process a batch of genes to extract PVC features.

        Args:
            genes: List of gene identifiers
            pvc_data: Dictionary mapping genes to variant data tensors
            max_variants: Maximum number of variants to consider

        Returns:
            Dictionary mapping genes to extracted feature vectors
        """
        features = {}
        for gene in genes:
            features[gene] = extract_pvc_features(gene, pvc_data, max_variants)
        return features
