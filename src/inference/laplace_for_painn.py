import torch
from tqdm import tqdm
import torch.nn as nn
from torch.func import hessian
import torch.nn.functional as F
from torch.linalg import LinAlgError
from torch.distributions import Normal
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.distributions.multivariate_normal import _precision_to_scale_tril


epsilon = 1e-8

def stable_softplus(x):
    return F.softplus(x) + epsilon


def painn_last_layer_and_nll(
    features: torch.Tensor,         # Features with shape (num_nodes, num_in_features)
    atoms: torch.Tensor,            # Atom type of each node with size (num_nodes,)
    param_vector: torch.Tensor,     # Parameter vector for last linear layer with shape (2*num_in_features + 2,)
    atom_references: torch.Tensor,  # Atomic references values with shape (num_atom_types, 1)
    scale: torch.Tensor,            # Scale value with shape ()
    shift: torch.Tensor,            # Shift value with shape ()
    y: torch.Tensor,                # Target with shape ()
) -> torch.Tensor:
    bias = param_vector[-2:]
    weight = param_vector[:-2].reshape(2, features.shape[1])

    # (num_nodes, num_in_features) -> (num_nodes, 2)
    out_features = F.linear(features, weight, bias)

    # Split into mean prediction and std prediction (num_nodes, 2) -> (num_nodes, 1), (num_nodes, 1)
    mean, std = torch.split(out_features, 1, dim=1)

    # Apply softplus to predicted standard deviations, (num_nodes, 1) -> (num_nodes, 1)
    std = stable_softplus(std)

    # Scale and shift means, (num_nodes, 1) -> (num_nodes, 1)
    mean = mean*scale + shift

    # Scale standard deviations, (num_nodes, 1) -> (num_nodes, 1)
    std = std*scale

    # Add atomic references values to mean, (num_nodes, 1) -> (num_nodes, 1)
    mean = mean + F.embedding(atoms, atom_references)

    # Compute variances, (num_nodes, 1) -> (num_nodes, 1)
    variance = std**2

    # Combine means and variances, (num_nodes, 1), (num_nodes, 1) -> (num_nodes, 2)
    output = torch.cat((mean, variance), dim=1)

    # Sum over atoms, (num_nodes, 2) -> (2,)
    output = output.sum(dim=0, keepdim=False)

    nll = F.gaussian_nll_loss(
        output[0],
        y,
        output[1],
        full=False,
        eps=0,
        reduction='none'
    )

    return nll


class LLLaplaceForPaiNN:
    def __init__(
        self,
        model: nn.Module,
        prior_precision: float = 1.,
        normalize_nll_hessian: bool = False,
    ) -> None:
        self.model = model
        self.prior_precision = prior_precision
        self.normalize_nll_hessian = normalize_nll_hessian
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        self.mean = parameters_to_vector(
            self.model.get_submodule('head.head.readout_network').parameters()
        )
        self.n_params = len(self.mean)
        self.atom_references = self.model.head.head.atom_refs
        self.scale = self.model.head.head.std
        self.shift = self.model.head.head.mean

        self._N = None
        self._H_nll = None
        self._H = None
        self._scale_tril = None
        self._posterior_covariance = None


    @property
    def N(self) -> int:
        if self._N is None:
            raise ValueError('No. datapoints not known. Call .fit() first.')
        return self._N


    @property
    def H_nll(self) -> torch.Tensor:
        """
        The Hessian of the negative log-likelihood at the MAP estimate.
        """
        if self._H_nll is None:
            raise ValueError('H_nll does not exist. Call .fit() first.')
        return self._H_nll


    @property
    def H(self) -> torch.Tensor:
        """
        The Hessian of the negative log joint distribution at the MAP estimate.
        This Hessian is the precision matrix of the Laplace approximation.
        """
        if self._H is None:
            prior_precision = self.prior_precision*torch.eye(
                self.n_params, device=self.device
            )
            H_nll = torch.clone(self.H_nll)
            if self.normalize_nll_hessian:
                H_nll /= self.N
            self._H = H_nll + prior_precision
        return self._H


    @property
    def scale_tril(self) -> torch.Tensor:
        """
        The Cholesky factor of the Laplace approximation's covariance matrix.
        """
        if self._scale_tril is None:
            self._scale_tril = _precision_to_scale_tril(self.H)
        return self._scale_tril


    @property
    def posterior_covariance(self):
        if self._posterior_covariance is None:
            self._posterior_covariance = self.scale_tril @ torch.t(self.scale_tril)
        return self._posterior_covariance


    def reset(self):
        """
        Resets self._H, self._scale_tril, and self._posterior_covariance to None
        """
        self._H = None
        self._scale_tril = None
        self._posterior_covariance = None


    def sample(self, n_samples: int = 100):
        eps = torch.randn(self.n_params, n_samples, device=self.device)
        return self.mean + torch.t(self.scale_tril @ eps)


    def fit(self, train_loader: torch.utils.data.DataLoader):
        self.model.eval()

        self._N = len(train_loader.dataset)
        H_nll = torch.zeros((self.n_params, self.n_params), device=self.device)
        compute_hessian = hessian(painn_last_layer_and_nll, argnums=2)

        for x, y in tqdm(train_loader):
            x, y = x.to(self.device), y.to(self.device)
            intermediate_output = self.model.backbone(x)

            for i, graph_idx in enumerate(intermediate_output['graph_indexes'].unique()):
                idx_vector = intermediate_output['graph_indexes'] == graph_idx
                features = intermediate_output['scalar_features'][idx_vector]
                atoms = intermediate_output['atoms'][idx_vector]
                yi = y[i,0]

                h = compute_hessian(
                    features, atoms, self.mean, self.atom_references, self.scale, self.shift, yi
                )
                H_nll += h.detach()

        self._H_nll = H_nll


    def optimize_prior_precision(
        self,
        val_loader: torch.utils.data.DataLoader,
        n_samples: int = 100,
        grid_size: int = 100,
        log_prior_prec_min: float = -4,
        log_prior_prec_max: float = 4,
        **kwargs,
    ) -> None:
        prior_precisions = torch.logspace(
            log_prior_prec_min, log_prior_prec_max, grid_size
        )
        losses = torch.ones_like(prior_precisions)*float('inf')

        # Precompute features
        self.model.eval()
        all_features = list()
        all_targets = list()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                all_features.append(self.model.backbone(x))                     # Dictionary with objects for batch
                all_targets.append(y.cpu())                                     # batch_size x num_targets
        all_targets = torch.cat(all_targets, dim=0)                             # num_datapoints x num_targets

        for i in tqdm(range(len(prior_precisions))):
            self.reset()
            self.prior_precision = prior_precisions[i]
            try:
                samples = self.sample(n_samples=n_samples)                      # num_samples x num_params
            except LinAlgError as e:
                print(f'The following error occured for prior precision {self.prior_precision}:')
                print(e)
                print('Setting loss for this precision to infinity.')
                continue

            outputs = list()
            for sample in samples:
                vector_to_parameters(
                    sample,
                    self.model.get_submodule('head.head.readout_network').parameters()
                )

                outputs_sample = list()
                with torch.no_grad():
                    for features in all_features:
                        outputs_sample.append(self.model.head(features).cpu())  # batch_size x num_outputs

                outputs_sample = torch.cat(outputs_sample, dim=0)               # num_datapoints x num_outputs
                outputs.append(outputs_sample)
            outputs = torch.stack(outputs, dim=0)                               # num_samples x num_datapoints x num_outputs

            # Compute ELPD
            S, N, K = outputs.shape
            model_means = outputs[:,:,0]
            model_vars = outputs[:,:,1]
            all_targets = all_targets.squeeze()
            log_densities = Normal(model_means, torch.sqrt(model_vars)).log_prob(all_targets)
            elpd = (
                -N*torch.log(torch.tensor(S))
                + torch.logsumexp(log_densities, dim=0).sum() 
            ).item() / N

            losses[i] = -elpd

        best_idx = torch.argmin(losses)
        best_prior_precision = prior_precisions[best_idx]
        best_loss = losses[best_idx]
        print('Prior precisions:')
        print(prior_precisions)
        print('Losses')
        print(losses)
        print('Best prior precision:')
        print(best_prior_precision)
        print('Best loss:')
        print(best_loss)

        self.reset()
        self.prior_precision = best_prior_precision.cpu().item()
        # Invoke computation of opbjects
        self.H
        self.scale_tril
        self.posterior_covariance


    def state_dict(self) -> dict:
        return {
            'N': self.N,
            'H_nll': self.H_nll.to('cpu'),
            'H': self.H.to('cpu'),
            'scale_tril': self.scale_tril.to('cpu'),
            'posterior_covariance': self.posterior_covariance.to('cpu'),
            'prior_precision': self.prior_precision,
        }


    def load_state_dict(self, state_dict: dict) -> None:
        self._N = state_dict['N']
        self._H_nll = state_dict['H_nll'].to(self.device)
        self._H = state_dict['H'].to(self.device)
        self._scale_tril = state_dict['scale_tril'].to(self.device)
        self._posterior_covariance = state_dict['posterior_covariance'].to(self.device)
        self.prior_precision = state_dict['prior_precision']
