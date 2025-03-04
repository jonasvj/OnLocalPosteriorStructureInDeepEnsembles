import os
import torch
import subprocess
import numpy as np
import pytorch_lightning as pl
from .qm9_utils import DataLoader
from torch_geometric.data import Data
from torch_geometric.datasets import QM9
from .alchemy_pyg import TencentAlchemyDataset
from typing import Optional, List, Union, Dict
from torch_geometric.transforms import BaseTransform


def download_alchemy_dataset(data_dir):
    alchemy_dir = os.path.join(data_dir, 'alchemy', 'raw')
    alchemy_zip = os.path.join(alchemy_dir, 'test_v20190730.zip')
    if not os.path.exists(alchemy_zip):
        subprocess.run(['mkdir', '-p', alchemy_dir])
        subprocess.run(['wget', '-P', alchemy_dir, 'https://alchemy.tencent.com/data/test_v20190730.zip'])
        subprocess.run(['unzip', '-d', alchemy_dir, alchemy_zip])


class GetTarget(BaseTransform):
    def __init__(self, target: Optional[int] = None) -> None:
        self.target = [target]
    
    def forward(self, data: Data) -> Data:
        if self.target is not None:
            data.y = data.y[:, self.target]
        return data


class QM9DataModule(pl.LightningDataModule):

    target_types = ['atomwise' for _ in range(19)]
    target_types[0] = 'dipole_moment'
    target_types[5] = 'electronic_spatial_extent'

    # Specify unit conversions (eV to meV).
    unit_conversion = {
        i: (lambda t: 1000*t) if i not in [0, 1, 5, 11, 16, 17, 18]
        else (lambda t: t)
        for i in range(19)
    }

    def __init__(
        self,
        target: int = 0,
        data_dir: str = 'data/',
        batch_size_train: int = 32,
        batch_size_inference: int = 32,
        num_workers: int = 0,
        splits: Union[List[int], List[float]] = [0.8, 0.1, 0.1],
        seed: int = 0,
        subset_size: Optional[int] = None,
        data_augmentation: bool = False, # Unused but here for compatibility
        name: str = 'qm9',
        ood: bool = False, # Unused but here for compatibility
        shift: bool = False, # Unused but here for compatibility
    ) -> None:
        super().__init__()
        self.target = target
        self.data_dir = data_dir
        self.batch_size_train = batch_size_train
        self.batch_size_inference = batch_size_inference
        self.num_workers = num_workers
        self.splits = splits
        self.seed = seed
        self.subset_size = subset_size
        self.data_augmentaion = data_augmentation
        self.name = name
        self.ood = ood

        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.ood_datasets = None


    def prepare_data(self) -> None:
        # Download data
        QM9(root=self.data_dir)

        if self.ood:
            download_alchemy_dataset(self.data_dir)
            TencentAlchemyDataset(
                root=os.path.join(self.data_dir, 'alchemy'),
                mode='test',
            )


    def setup(self, stage: Optional[str] = None) -> None:
        dataset = QM9(root=self.data_dir, transform=GetTarget(self.target))

        # Shuffle dataset
        rng = np.random.default_rng(seed=self.seed)
        dataset = dataset[rng.permutation(len(dataset))]

        # Subset dataset
        if self.subset_size is not None:
            dataset = dataset[:self.subset_size]
        
        # Split dataset
        if all([type(split) == int for split in self.splits]):
            split_sizes = self.splits
        elif all([type(split) == float for split in self.splits]):
            split_sizes = [int(len(dataset) * prop) for prop in self.splits]

        split_idx = np.cumsum(split_sizes)
        self.data_train = dataset[:split_idx[0]]
        self.data_val = dataset[split_idx[0]:split_idx[1]]
        self.data_test = dataset[split_idx[1]:]

        if self.ood:
            alchemy = TencentAlchemyDataset(
                root=os.path.join(self.data_dir, 'alchemy'),
                mode='test',
            )
            subset_indices = torch.randperm(
                len(alchemy),
                generator=torch.Generator().manual_seed(self.seed)
            )[:len(self.data_test)]
            alchemy = alchemy.index_select(subset_indices)

            self.ood_datasets = {
                'alchemy': alchemy
            }


    def get_target_stats(self, remove_atom_refs=False, divide_by_atoms=False):
        atom_refs = self.data_train.atomref(self.target)

        ys = list()
        for batch, _ in self.train_dataloader(shuffle=False):
            y = batch.y.clone()
            if remove_atom_refs and atom_refs is not None:
                y.index_add_(
                    dim=0, index=batch.batch, source=-atom_refs[batch.z]
                )
            if divide_by_atoms:
                _, num_atoms  = torch.unique(batch.batch, return_counts=True)
                y = y / num_atoms.unsqueeze(-1)
            ys.append(y)

        y = torch.cat(ys, dim=0)
        return y.mean(), y.std(), atom_refs


    def train_dataloader(self, shuffle=True) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size_train,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=True,
        )


    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size_inference,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )


    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size_inference,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )


    def ood_dataloaders(self) -> Dict[str, DataLoader]:
        return {
            dataset_name: DataLoader(
                dataset,
                batch_size=self.batch_size_inference,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True,
            )
            for dataset_name, dataset in self.ood_datasets.items()
        }
