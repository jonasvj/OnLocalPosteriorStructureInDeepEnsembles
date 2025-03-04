import os
import torch
import numpy as np
import pytorch_lightning as pl
from torchtext import datasets
import torchvision.transforms.v2 as transforms
from typing import Optional, Tuple, Sequence, Dict
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import TensorDataset


class SST2DataModule(pl.LightningDataModule):
    """"
    The SST2 dataset wrapped in a PyTorch Lightning data module.
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
        name: str = "SST2",
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


    def split_data(
        self,
        num_train: int = 67349,
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


    def setup(self, stage: Optional[str] = None) -> None:
        """
        Does all the necessary data preprocessing.
        """
        # Split training data into training and validation data
        #train_indices, val_indices = self.split_data(num_train=67349)
        

        #TODO: How to preprocess from here??
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

        self.train_dataset = load_dataset("stanfordnlp/sst2", split="train")
        self.val_dataset = load_dataset("stanfordnlp/sst2", split="validation")

        #We get the test set from SetFit because there they actually have labels (and they are the same)
        self.test_dataset = load_dataset("SetFit/sst2", split="test")
        #This could be significantly sped up by making collated batches with max lengths per batch rather than across dataset...
        #Max length in dataset is 
        def preprocess_setfit_function(examples):
            return self.tokenizer(examples["text"], truncation=True, max_length=268, padding="max_length")
        def preprocess_stanfordnlp_function(examples):
            return self.tokenizer(examples["sentence"], truncation=True, max_length=268, padding="max_length")

        self.data_train = self.train_dataset.map(preprocess_stanfordnlp_function, batched=True)
        self.data_val = self.val_dataset.map(preprocess_stanfordnlp_function, batched=True)
        self.data_test = self.test_dataset.map(preprocess_setfit_function, batched=True)

        #Want dataloaders to output x,y but where x=("input_ids", "attention_mask")
        def create_torch_dataset(dataset):
            return TensorDataset(torch.stack((dataset['input_ids'], dataset['attention_mask']),dim=1), dataset['label'])

        self.data_train.set_format("torch")
        self.data_val.set_format("torch")
        self.data_test.set_format("torch")

        self.data_train = create_torch_dataset(self.data_train)
        self.data_val = create_torch_dataset(self.data_val)

        #train_indices, val_indices = self.split_data(num_train=67349)

        #self.data_val = Subset(self.data_train, indices=val_indices)
        #self.data_train = Subset(self.data_train, indices=train_indices)
        self.data_test = create_torch_dataset(self.data_test)

        if self.ood:
            yahoo = load_dataset(
                'ugursa/Yahoo-Finance-News-Sentences',
                split='train',
            )
            yahoo = yahoo.map(preprocess_setfit_function, batched=True)
            yahoo.set_format("torch")
            yahoo = create_torch_dataset(yahoo)
            subset_indices = torch.randperm(
                len(yahoo),
                generator=torch.Generator().manual_seed(self.seed)
            )[:len(self.data_test)]
            yahoo = Subset(yahoo, indices=subset_indices)
            self.ood_datasets = {
                'yahoo': yahoo,
            }


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
