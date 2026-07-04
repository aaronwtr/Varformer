"""Synchronized batch sampler and dataloader for multi-modal training data."""
import torch

from torch.utils.data import Dataset, BatchSampler, Sampler
from typing import Dict, List, Iterator, Union, Iterable

from varformer.data.datasets import MultiModalData


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
