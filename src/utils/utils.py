import os
import pyro
import torch
import pandas as pd
import torch.nn as nn
from typing import List
from pytorch_lightning import seed_everything
from src.data.cifar10 import *
from src.data.cifar100 import *
from src.data.qm9 import QM9DataModule
from src.data.SST2 import SST2DataModule
from omegaconf.errors import ConfigKeyError

SEEDS = list(range(1, 31))

DATA_NAMES = {
    'cifar10': 'CIFAR-10',
    'cifar100': 'CIFAR-100',
    'qm9': 'QM9',
    'sst2': 'SST-2',
}

INFERENCE_NAMES = {
    'MAP': 'DE',
    'SWAG': 'SWAG',
    'LastLayerLaplace': 'LLLA',
    'PosteriorRefinedLastLayerLaplace': 'LLLA-NF',
    'DeepEnsemble': 'DE',
    'MultiSWAG': 'SWAG',
    'MoLA': 'LLLA',
    'MoFlowLA': 'LLLA-NF',
    'SWA': 'SWA',
    'MultiSWA': 'SWA',
    'SampleSWAG': 'Samples',
    'MultiSampleSWAG': 'Samples',
    'IVONFromScratch': 'IVON',
    'MultiIVONFromScratch': 'IVON',
    'MonteCarloDropout': 'MCDO',
    'MultiMCDO': 'MCDO',
}

INFERENCE_SYMBOLS = {
    'DE': 'o',
    'SWA': 'P',
    'LLLA': '^',
    'LLLA-NF': 'p',
    'LA-NF': 'p', 
    'SWAG': 'X',
    'IVON': 'd',
    'MCDO': '*',
}

K_COLORS = {
    1: 'C0',
    2: 'C1',
    5: 'C2',
    10: 'C3',
    20: 'C4',
}

DATA_COLORS = {
    'CIFAR-10': 'C5',
    'CIFAR-10 (WRN-16-4)': 'C5',
    'CIFAR-10 (WRN-28-10)': 'C6',
    'CIFAR-100': 'C7',
    'CIFAR-100 (WRN-16-4)': 'C7',
    'SST-2': 'C8',
    'QM9': 'C9'
}


def set_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    pyro.set_rng_seed(seed)

    torch.backends.cudnn.benchmark = False      # Ensures algorithm selection is deterministic
    torch.backends.cudnn.deterministic = True   # Ensures the algorithm itself is deterministic
    torch.use_deterministic_algorithms(True, warn_only=True)

    seed_everything(seed, workers=True)

def get_data_batch(dataset: str):
    if dataset == "cifar10":
        dm = CIFAR10DataModule(data_dir="data/")
    elif dataset == "cifar100":
        dm = CIFAR100DataModule(data_dir="data/")
    elif dataset == 'qm9':
        dm = QM9DataModule(data_dir="data/")
    elif dataset == "SST2":
        dm = SST2DataModule(data_dir="data/")
    dm.prepare_data()
    dm.setup()
    
    train_loader = dm.train_dataloader()

    X_train, Y_train = next(iter(train_loader))

    return X_train, Y_train


def interpolation_loss(
    preds: torch.Tensor,
    y: torch.Tensor,
    reduction: str = 'sum',
    lam: float = 1.,
    y_var: float = 1.,
) -> torch.Tensor:
    mean = preds[:, 0]
    var = preds[:, 1]
    y = y.squeeze()
    mse = nn.functional.mse_loss(mean, y, reduction=reduction)
    nll = y_var*nn.functional.gaussian_nll_loss(
        mean, y, var, full=False, eps=0., reduction=reduction
    )
    return lam*mse + (1 - lam)*nll


def lam_schedule(step: int, type: str = 'interpolation'):
    if type == 'interpolation':
        return interpolation_lam_schedule(step)
    elif type == 'const':
        return const_lam_schedule(step)


def const_lam_schedule(step: int, const: float = 1.):
    return const


def interpolation_lam_schedule(
    step: int,
    num_warmup_steps: int = 50000, 
    num_interpolation_steps: int = 50000
):
    if step < num_warmup_steps:
        return 1.
    elif step < num_warmup_steps + num_interpolation_steps:
        return 1. - (step - num_warmup_steps) / (num_interpolation_steps - 1.)
    else:
        return 0.


def load_stats(directories: List[str], model_key: str='wrn-16-4', data_key: str = 'cifar10'):
    # Load results
    results = list()
    common_keys = set()
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        if data_key not in directory:
            continue

        aug = True if 'aug_true' in directory else False
        norm = 'bn' if '_bn' in directory else 'frn'
        norm = norm if 'wrn' in model_key else None

        if 'cifar100' in directory:
            data = DATA_NAMES['cifar100']
        elif 'cifar10' in directory:
            data = DATA_NAMES['cifar10']
        elif 'qm9' in directory:
            data = DATA_NAMES['qm9']
        elif 'sst2' in directory:
            data = DATA_NAMES['sst2']
        else:
            data = None

        files = os.listdir(directory)
        for file in sorted(files):

            if model_key not in file:
                continue

            stats_dict = torch.load(os.path.join(directory, file))

            for row in stats_dict['results']:
                if all([int(seed) in SEEDS for seed in row['model_seeds'].split(',')]):
                    inference = INFERENCE_NAMES[row['inference']]
                    if inference == 'LLLA-NF':
                        num_transforms = file.split('_num-transform=')[-1].split('_')[0]
                        inference = inference + '-' + num_transforms

                    row['inference'] = inference 
                    row['aug'] = aug
                    row['norm'] = norm
                    row['data'] = data
                    try:
                        row['cov_scale'] = stats_dict['cfg']['covariance_scale_factor']
                    except ConfigKeyError:
                        row['cov_scale'] = 1
                    results.append(row)

                    if len(common_keys) == 0:
                        common_keys.update(row.keys())
                    else:
                        common_keys.intersection_update(row.keys())

    common_keys = list(common_keys)
    common_keys.remove('split')

    df = pd.DataFrame(results)
    if len(df.columns) - 1 != len(common_keys):
        df = df.groupby(common_keys, dropna=False).sum()
    df = df.reset_index()

    return df

def unfreeze_pretrained_backbone(model):
    module_dict = dict(model.named_parameters())
    for key in module_dict.keys():
        module_dict[key].requires_grad = True
