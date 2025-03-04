import os
import torch
import numpy as np
import pytorch_lightning as pl
from torchvision import datasets
import torchvision.transforms.v2 as transforms
from typing import Optional, Tuple, Sequence, Dict
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset


def load_cifar10_corrupted(cifar10_corrupted_dir, transform=None):
    """
    Adapted from:
    https://github.com/MJordahn/Decoupled-Layers-for-Calibrated-NNs/blob/2c26b8c2307752175cd9e92b2a0d59145f90851b/src/utils/eval_utils.py#L113
    """
    files = sorted(os.listdir(cifar10_corrupted_dir))
    files.remove('labels.npy')
    labels = np.load(os.path.join(cifar10_corrupted_dir, 'labels.npy'))

    corruption_levels = list(range(5))
    datasets = {f'cifar10_c_level_{c}': list() for c in corruption_levels}

    for file in files:
        images = np.load(os.path.join(cifar10_corrupted_dir, file))

        for c_level, dataset_name in enumerate(sorted(datasets.keys())):
            idx_start, idx_end = c_level*10000, (c_level + 1)*10000
            datasets[dataset_name].append(
                (images[idx_start:idx_end], labels[idx_start:idx_end])
            )

    for dataset_name, data_list in datasets.items():
        images = np.concatenate([data[0] for data in data_list], axis=0)
        labels = np.concatenate([data[1] for data in data_list], axis=0)

        # Reshape images from N x H x W x C to N x C x H x W
        images = np.moveaxis(images, -1, 1)
        
        # Convert to tensors
        images = torch.from_numpy(images)
        labels = torch.from_numpy(labels)

        if transform is not None:
            images = torch.stack([transform(image) for image in images], dim=0)

        # Create dataset
        datasets[dataset_name] = TensorDataset(images, labels.to(torch.int64))

    return datasets
    

class CIFAR10DataModule(pl.LightningDataModule):
    """"
    The CIFAR-10 dataset wrapped in a PyTorch Lightning data module.
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
        name: str = "cifar10",
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
        datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        datasets.CIFAR10(root=self.data_dir, train=False, download=True)

        if self.ood:
            datasets.SVHN(root=self.data_dir, split='test', download=True)
            datasets.CIFAR100(root=self.data_dir, train=False, download=True)

        if self.shift:
            if not os.path.isdir(os.path.join(self.data_dir, 'CIFAR-10-C')):
                raise ValueError(
                    f'CIFAR-10-C data folder not present. Download the '
                    f'CIFAR-10-C data from https://zenodo.org/records/2535967 '
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
        data_train = datasets.CIFAR10(
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
                datasets.CIFAR10(
                    root=self.data_dir, train=True, transform=train_transforms
                ),
                indices=torch.Tensor(range(self.batch_size_train)).int(),
            )
        else:
            self.data_train = Subset(
                datasets.CIFAR10(
                    root=self.data_dir, train=True, transform=train_transforms
                ),
                indices=train_indices,
            )
        self.data_val = Subset(
            datasets.CIFAR10(
                root=self.data_dir, train=True, transform=test_transforms
            ),
            indices=val_indices,
        )
        self.data_test = datasets.CIFAR10(
            root=self.data_dir, train=False, transform=test_transforms
        )

        if self.ood:
            svhn = datasets.SVHN(
                root=self.data_dir,
                split='test',
                transform=test_transforms,
            )
            cifar100 = datasets.CIFAR100(
                root=self.data_dir,
                train=False,
                transform=test_transforms
            )
            self.ood_datasets = {
                'svhn_cifar100': ConcatDataset([svhn, cifar100])
            }

        if self.shift:
            self.shift_datasets = load_cifar10_corrupted(
                os.path.join(self.data_dir, 'CIFAR-10-C'),
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
