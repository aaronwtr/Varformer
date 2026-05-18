"""Gene-level feature extraction from OpenTargets and population exome data."""
import os
import pickle as pkl

import pandas as pd

from typing import Optional

from varformer.data.splits import load_combined_labels, combine_features_and_labels, get_labels
from varformer.data.features.base import BaseFeatures


# data preprocessing
class GeneCharacterisationPreprocessor(BaseFeatures):
    """
    This class loads and combines the different data sources into a single feature matrix to be fed into our model.
    """

    def __init__(self, config, base: Optional[BaseFeatures] = None):
        print("Gene Characterisation Preprocessor is booting up...")
        if base is not None:
            # Adopt the BaseFeatures state without re-running its __init__
            self.__dict__.update(base.__dict__)
        else:
            super().__init__(config)

        features_dir = self.config['paths']['FEATURES_DIR']
        population = self.config['hyperparameters']['population']

        self.features = self.load_opentargets_features()

        self.features = self.features[self.features['targetId'].isin(self.pop_data['Gene'])]
        nan_percentages = self.features.isna().mean() * 100
        high_nan_features = nan_percentages[nan_percentages > 99].index.tolist()
        if high_nan_features:
            print(f"Removing features with only NaN values: {high_nan_features}")
            self.features = self.features.drop(columns=high_nan_features)

        # check feature statistics in the features attribute here

        tissue_columns = [col for col in self.features.columns if 'tissueDistribution' in col]
        for col in tissue_columns:
            self.features[col] = self.features[col].fillna(-0.5)

        self.features = self.features.fillna(0)

        self.ensg_ids = self.features["targetId"]

        # Ground truth
        self.features = self.features.rename(columns={'maxClinicalTrialPhase': 'target'})
        self.features['target'] = self.features['target'].apply(lambda x: 1.0 if x >= 0.75 else 0.0)
        self.features = self.features[[col for col in self.features if col != 'target'] + ['target']]
        self.ot_targets = self.features[['targetId', 'target']]
        self.target = load_combined_labels(self.ot_targets, self.config)

        # Combine features and target
        self.labels_dict = get_labels(self.ensg_ids, self.target)
        self.full_data = combine_features_and_labels(self.ensg_ids, self.features, self.target)
        self.full_data.set_index('targetId', inplace=True)
        # feature statistics can be checked here!

        self.ce_data = self.full_data
        self.num_features = len(self.full_data.columns) - 1

        self.data = self.full_data
        self.labels = self.labels_dict

        # Create population directory if it doesn't exist
        os.makedirs(f'{features_dir}/{population}', exist_ok=True)

        # Save the GC features (self.data contains the final feature matrix)
        gc_features_path = f'{features_dir}/{population}/gene_characterisation_features.pkl'
        with open(gc_features_path, 'wb') as f:
            pkl.dump(self.data, f)

    def load_ground_truth(self):
        """
        Load the ground truth data.
        """
        return self.datasets["FDA Approved Drug Targets"]

    def load_opentargets_features(self):
        feature_path = self.config['paths']['OT_PATH']
        ot_df = pd.read_pickle(feature_path)
        cols_rm = ["isInMembrane", "isSecreted", "isCancerDriverGene", "tissueSpecificity"]
        cols = ot_df.columns
        cols_to_keep = [col for col in cols if col not in cols_rm]
        ot_df = ot_df[cols_to_keep]
        return ot_df

