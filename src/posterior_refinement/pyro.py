"""
Adapted from: https://github.com/runame/laplace-refinement
"""
import math

import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional

from src.utils.posterior_utils import *


epsilon = 1e-8

def stable_softplus(x):
    return F.softplus(x) + epsilon


class Model:

    def __init__(self, n_params, n_data, prior_prec=10, cuda=False,
                 proj_mat=None, base_dist=None, diag=False):
        self.uuid = np.random.randint(low=0, high=10000, size=1)[0]

        self.n_data = n_data
        self.prior_mean = torch.zeros(n_params, device='cuda' if cuda else 'cpu')
        self.prior_std = math.sqrt(1/prior_prec)
        self.cuda = cuda

        if proj_mat is not None:
            self.A = proj_mat
            self.k, self.d = self.A.shape

            self.prior_mean_proj = torch.zeros(self.k)
            self.prior_Cov_proj = self.prior_std**2 * self.A @ self.A.T

            if base_dist is not None:
                self.base_dist = base_dist

                self.base_mean_proj = self.A @ self.base_dist.mean

                if not diag:
                    self.base_Cov_proj = self.A @ self.base_dist.covariance_matrix @ self.A.T
                else:
                    self.base_Cov_proj = self.A * (self.base_dist.scale**2)[None, :] @ self.A.T

                self.base_dist_proj = dist.MultivariateNormal(self.base_mean_proj, self.base_Cov_proj)

        if cuda:
            self.prior_mean = self.prior_mean.cuda()

            if proj_mat is not None and base_dist is not None:
                self.prior_mean_proj = self.prior_mean_proj.cuda()
                self.prior_Cov_proj = self.prior_Cov_proj.cuda()

    def model(self, X, y=None):
        raise NotImplementedError()

    def model_subspace(self, X, y=None):
        raise NotImplementedError()


class RegressionModel(Model):

    def __init__(self, get_net, n_data, prior_prec=10, log_noise=torch.tensor(1.), cuda=False, proj_mat=None, base_dist=None):
        n_params = sum(p.numel() for p in self.get_net().parameters())
        super().__init__(n_params, n_data, prior_prec, cuda, proj_mat, base_dist)
        self.get_net = get_net
        self.noise = F.softplus(log_noise)

    def model(self, X, y=None):
        # Sample params from the prior
        theta = pyro.sample('theta', dist.Normal(self.prior_mean, self.prior_std).to_event(1))

        # Put the sample into the net
        net = self.get_net()
        vector_to_parameters_backpropable(theta, net)
        f_X = net(X).squeeze()

        # Likelihood
        if y is not None:
            # Training
            with pyro.plate('data', size=self.n_data, subsample=y.squeeze()):
                pyro.sample('obs', dist.Normal(f_X, self.noise), obs=y.squeeze())
        else:
            # Testing
            pyro.sample('obs', dist.Normal(f_X, self.noise))

    def model_subspace(self, X, y=None, full_batch=False):
        # Sample params from the prior on low-dim, then project it to high-dim
        z = pyro.sample('z', dist.MultivariateNormal(self.prior_mean_proj, self.prior_Cov_proj))
        theta = self.A.T @ z

        # Put the sample into the net
        net = self.get_net()
        vector_to_parameters_backpropable(theta, net)
        f_X = net(X).squeeze()

        # Likelihood
        if y is not None:
            # Training
            with pyro.plate('data', size=self.n_data, subsample=y.squeeze()):
                pyro.sample('obs', dist.Normal(f_X, self.noise), obs=y.squeeze())
        else:
            # Testing
            pyro.sample('obs', dist.Normal(f_X, self.noise))


class ClassificationModel(Model):

    def __init__(self, get_net, n_data, prior_prec=10, cuda=False, proj_mat=None, base_dist=None, diag=True):
        self.get_net = get_net

        n_params = sum(p.numel() for p in self.get_net().parameters())
        super().__init__(n_params, n_data, prior_prec, cuda, proj_mat, base_dist, diag)

    def model(self, X, y=None, full_batch=False):
        # Sample params from the prior
        theta = pyro.sample('theta', dist.Normal(self.prior_mean, self.prior_std).to_event(1))

        # Put the sample into the net
        net = self.get_net()

        if self.cuda:
            net.cuda()

        
        vector_to_parameters_backpropable(theta, net)
        f_X = net(X)

        # Likelihood
        if y is not None:
            subsample = None if full_batch else y.squeeze()

            with pyro.plate('data', size=self.n_data, subsample=subsample):
                pyro.sample('obs', dist.Categorical(logits=f_X), obs=y.squeeze())

        return f_X

    def model_subspace(self, X, y=None, full_batch=False):
        # Sample params from the prior on low-dim, then project it to high-dim
        z = pyro.sample('z', dist.MultivariateNormal(self.prior_mean_proj, self.prior_Cov_proj))
        theta = self.A.T @ z

        # Put the sample into the net
        net = self.get_net()

        if self.cuda:
            net.cuda()

        vector_to_parameters_backpropable(theta, net)
        f_X = net(X)

        # Likelihood
        if y is not None:
            subsample = None if full_batch else y.squeeze()

            with pyro.plate('data', size=self.n_data, subsample=subsample):
                pyro.sample('obs', dist.Categorical(logits=f_X), obs=y.squeeze())

        return f_X


class ClassificationModelLL(Model):

    def __init__(self, n_data, n_features, n_classes, feature_extractor, prior_prec=10, cuda=False, proj_mat=None, base_dist=None, diag=False):
        n_params = n_features*n_classes + n_classes  # weights and biases
        super().__init__(n_params, n_data, prior_prec, cuda, proj_mat, base_dist, diag)

        self.n_features = n_features
        self.n_classes = n_classes
        self.feature_extractor = feature_extractor

    def model(self, X, y=None, full_batch=False, X_is_features=False):
        # Sample params from the prior
        theta = pyro.sample('theta', dist.Normal(self.prior_mean, self.prior_std).to_event(1))
        f_X = self._forward(X, theta, X_is_features)

        # Likelihood
        if y is not None:
            subsample = None if full_batch else y.squeeze()

            with pyro.plate('data', size=self.n_data, subsample=subsample):
                pyro.sample('obs', dist.Categorical(logits=f_X), obs=y.squeeze())

        return f_X

    def _forward(self, X, theta, X_is_features=False):
        # Make it compatible with PyTorch's parameters vectorization that Laplace uses
        W = theta[:self.n_features*self.n_classes].reshape(self.n_classes, self.n_features)
        b = theta[-self.n_classes:]

        if X_is_features:
            phi_X = X
        else:
            with torch.no_grad():
                phi_X = self.feature_extractor(X)

        return phi_X @ W.T + b  # Transpose following nn.Linear


class PaiNNModelLL(Model):

    def __init__(
        self,
        n_data: int,
        n_features: int,
        feature_extractor: nn.Module,
        scale: torch.Tensor, 
        shift: torch.Tensor,
        atom_references: torch.Tensor,
        prior_prec: float = 10.,
        cuda: bool = False,
        proj_mat: Optional[torch.Tensor] = None,
        base_dist: Optional[dist.MultivariateNormal] = None,
        diag: bool = False,
    ):
        n_params = n_features*2 + 2  # weights and biases
        super().__init__(n_params, n_data, prior_prec, cuda, proj_mat, base_dist, diag)

        self.n_features = n_features
        self.feature_extractor = feature_extractor
        self.scale = scale
        self.shift = shift
        self.atom_references = atom_references


    def model(self, X, y=None, full_batch=False, X_is_features=False):
        # Sample params from the prior
        theta = pyro.sample('theta', dist.Normal(self.prior_mean, self.prior_std).to_event(1))
        f_X = self._forward(X, theta, X_is_features)

        # Likelihood
        if y is not None:
            subsample = None if full_batch else y.squeeze()

            with pyro.plate('data', size=self.n_data, subsample=subsample):
                mean = f_X[:,0]
                variance = f_X[:,1]
                pyro.sample(
                    'obs',
                    dist.Normal(loc=mean, scale=torch.sqrt(variance)),
                    obs=y.squeeze()
                )

        return f_X


    def _forward(self, X, theta, X_is_features=False):
        bias = theta[-2:]
        weight = theta[:-2].reshape(2, self.n_features)

        if X_is_features:
            phi_X = X
        else:
            with torch.no_grad():
                phi_X = self.feature_extractor(X)
    
        features = phi_X['scalar_features'] # (num_nodes, num_features)
        atoms = phi_X['atoms'] # (num_nodes,)
        graph_indexes = phi_X['graph_indexes'] # (num_nodes,)
        num_graphs = torch.unique(graph_indexes).shape[0]

        # (num_nodes, num_in_features) -> (num_nodes, 2)
        out_features = F.linear(features, weight, bias)

        # Split into mean prediction and std prediction (num_nodes, 2) -> (num_nodes, 1), (num_nodes, 1)
        mean, std = torch.split(out_features, 1, dim=1)

        # Apply softplus to predicted standard deviations, (num_nodes, 1) -> (num_nodes, 1)
        std = stable_softplus(std)

        # Scale and shift means, (num_nodes, 1) -> (num_nodes, 1)
        mean = mean*self.scale + self.shift

        # Scale standard deviations, (num_nodes, 1) -> (num_nodes, 1)
        std = std*self.scale

        # Add atomic references values to mean, (num_nodes, 1) -> (num_nodes, 1)
        mean = mean + F.embedding(atoms, self.atom_references)

        # Compute variances, (num_nodes, 1) -> (num_nodes, 1)
        variance = std**2

        # Combine means and variances, (num_nodes, 1), (num_nodes, 1) -> (num_nodes, 2)
        output = torch.cat((mean, variance), dim=1)

        # Sum contributions for each graph
        output_per_graph = torch.zeros(
            (num_graphs, 2),
            device=output.device
        )
        output_per_graph.index_add_(
            dim=0,
            index=graph_indexes,
            source=output,
        )

        return output_per_graph
