"""
Adapted from: https://github.com/runame/laplace-refinement
"""
import functools
import os
import time
from pathlib import Path

import numpy as np
import pyro.distributions as dist
import scipy.stats as st
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.models as torch_models
from laplace.curvature import AsdlEF, AsdlGGN, BackPackEF, BackPackGGN
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from sklearn.metrics import mean_squared_error, roc_auc_score
from torch.nn.utils import parameters_to_vector

def vector_to_parameters_backpropable(vec, net):
    # Pointer for slicing the vector for each parameter
    pointer = 0
    for mod in net.children():
        if isinstance(mod, torch.nn.Linear) or isinstance(mod, torch.nn.Conv2d):
            weight_size, bias_size = mod.weight.shape, mod.bias.shape
            weight_numel, bias_numel = mod.weight.numel(), mod.bias.numel()

            del mod.weight
            del mod.bias

            mod.weight = vec[pointer:pointer+weight_numel].reshape(weight_size)
            pointer += weight_numel

            mod.bias = vec[pointer:pointer+bias_numel].reshape(bias_size)
            pointer += bias_numel
