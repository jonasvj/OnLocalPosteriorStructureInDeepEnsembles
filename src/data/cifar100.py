import os
import torch
import numpy as np
import pytorch_lightning as pl
from torchvision import datasets
import torchvision.transforms.v2 as transforms
from typing import Optional, Tuple, Sequence, Dict
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset


def load_cifar100_corrupted(cifar100_corrupted_dir, transform=None):
    raise NotImplementedError


class CIFAR100DataModule(pl.LightningDataModule):
    """"
    The CIFAR-100 dataset wrapped in a PyTorch Lightning data module.
    """
    def __init__(
        self,
        data_dir: str = 'data/',
        batch_size_train: int = 128,
        batch_size_inference: int = 128,
        data_augmentation: bool = False,
        num_workers: int = 0,
        num_val: int = 0,
        seed: int = 0,
        ood: bool = False,
        shift: bool = False,
        name: str = "cifar100",
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size_train = batch_size_train
        self.batch_size_inference = batch_size_inference
        self.data_augmentation = data_augmentation
        self.num_workers = num_workers
        self.num_val = num_val
        self.seed = seed
        self.ood = ood
        self.shift = shift
        self.name = name
        self.debug = debug
        if debug:
            self.num_workers = 0

        self.persistent_workers = True if self.num_workers > 0 else False

        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.ood_datasets = None
        self.shift_datasets = None


    def prepare_data(self) -> None:
        """
        Downloads the data.
        """
        datasets.CIFAR100(root=self.data_dir, train=True, download=True)
        datasets.CIFAR100(root=self.data_dir, train=False, download=True)

        if self.ood:
            datasets.SVHN(root=self.data_dir, split='test', download=True)
            datasets.CIFAR10(root=self.data_dir, train=False, download=True)

        if self.shift:
            if not os.path.isdir(os.path.join(self.data_dir, 'CIFAR-100-C')):
                raise ValueError(
                    f'CIFAR-100-C data folder not present. Download the '
                    f'CIFAR-100-C data from https://zenodo.org/records/3555552 '
                    f'and unpack it in {self.data_dir}.'
                )


    def split_data(
        self,
        num_train: int = 50000,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Splits training indices into train and validation indices.
        """
        indices = torch.randperm(
            num_train,
            generator=torch.Generator().manual_seed(self.seed)
        )
        val_indices = indices[:self.num_val]
        train_indices = indices[self.num_val:]

        return train_indices, val_indices


    def compute_channel_stats(
        self,
        train_indices: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the mean and standard deviation of each channel in the 
        training images.
        """
        data_train = datasets.CIFAR100(
            root=self.data_dir,
            train=True,
            transform=transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True)
            ]),
        )
        data_train = Subset(data_train, indices=train_indices)
        
        X_train = torch.stack([img[0] for img in data_train])
        mean = X_train.mean(dim=(0,2,3))
        std = X_train.std(dim=(0,2,3))

        return mean, std


    def setup(self, stage: Optional[str] = None) -> None:
        """
        Does all the necessary data preprocessing.
        """
        # Split training data into training and validation data
        train_indices, val_indices = self.split_data(num_train=50000)
        
        # Compute channel means and standard deviations for training data
        mean, std = self.compute_channel_stats(train_indices=train_indices)

        # Transformations for training and testing
        test_transforms = [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std),
        ]
        if self.data_augmentation:
            train_transforms = [
                transforms.ToImage(), 
                transforms.ToDtype(torch.uint8, scale=True),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean, std),
            ]
        else:
            train_transforms = test_transforms[:]

        train_transforms = transforms.Compose(train_transforms)
        test_transforms = transforms.Compose(test_transforms)
        
        if self.debug:
            self.data_train = Subset(
                datasets.CIFAR100(
                    root=self.data_dir, train=True, transform=train_transforms
                ),
                indices=torch.Tensor(range(self.batch_size_train)).int(),
            )
        else:
            self.data_train = Subset(
                datasets.CIFAR100(
                    root=self.data_dir, train=True, transform=train_transforms
                ),
                indices=train_indices,
            )
        self.data_val = Subset(
            datasets.CIFAR100(
                root=self.data_dir, train=True, transform=test_transforms
            ),
            indices=val_indices,
        )
        self.data_test = datasets.CIFAR100(
            root=self.data_dir, train=False, transform=test_transforms
        )

        if self.ood:
            svhn = datasets.SVHN(
                root=self.data_dir,
                split='test',
                transform=test_transforms,
            )
            cifar10 = datasets.CIFAR10(
                root=self.data_dir,
                train=False,
                transform=test_transforms
            )
            self.ood_datasets = {
                'svhn_cifar10': ConcatDataset([svhn, cifar10])
            }

        if self.shift:
            self.shift_datasets = load_cifar100_corrupted(
                os.path.join(self.data_dir, 'CIFAR-100-C'),
                transform=test_transforms
            )


    def train_dataloader(self, shuffle=True) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size_train,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=True,
            persistent_workers=self.persistent_workers,
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
    

    def shift_dataloaders(self) -> Dict[str, DataLoader]:
        return {
            dataset_name: DataLoader(
                dataset,
                batch_size=self.batch_size_inference,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True,
            )

            for dataset_name, dataset in self.shift_datasets.items()
        }
