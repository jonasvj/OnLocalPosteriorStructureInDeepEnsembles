import math
import torch
from laplace import FullLLLaplace, DiagLLLaplace
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class FullLLLaplaceWrapper(FullLLLaplace):
    def state_dict(self) -> dict:
        self._check_H_init()
        laplace_state_dict = {
            'mean': self.mean,
            'H': self.H,
            'loss': self.loss,
            'prior_mean': self.prior_mean,
            'prior_precision': self.prior_precision,
            'sigma_noise': self.sigma_noise,
            'n_data': self.n_data,
            'n_outputs': self.n_outputs,
            'likelihood': self.likelihood,
            'temperature': self.temperature,
            'cls_name': self.__class__.__name__,
            'data' : getattr(self, 'data', None),  # None if not present
        }
        return laplace_state_dict


    def load_state_dict(self, state_dict:dict):

        self.data = state_dict['data']

        self.mean = state_dict['mean']
        self.H = state_dict['H']
        self.loss = state_dict['loss']
        self.prior_mean = state_dict['prior_mean']
        self.prior_precision = state_dict['prior_precision']
        self.sigma_noise = state_dict['sigma_noise']
        self.n_data = state_dict['n_data']
        self.n_outputs = state_dict['n_outputs']
        setattr(self.model, 'output_size', self.n_outputs)
        self.likelihood = state_dict['likelihood']
        self.temperature = state_dict['temperature']

        #params = parameters_to_vector(self.model.head.parameters()).detach()
        #self.n_params = len(params)
        #self.n_layers = len(list(self.model.head.parameters()))

class DiagLLLaplaceWrapper(DiagLLLaplace):
    def state_dict(self) -> dict:
        self._check_H_init()
        laplace_state_dict = {
            'mean': self.mean,
            'H': self.H,
            'loss': self.loss,
            'prior_mean': self.prior_mean,
            'prior_precision': self.prior_precision,
            'sigma_noise': self.sigma_noise,
            'n_data': self.n_data,
            'n_outputs': self.n_outputs,
            'likelihood': self.likelihood,
            'temperature': self.temperature,
            'cls_name': self.__class__.__name__,
            'data' : getattr(self, 'data', None),  # None if not present
        }
        return laplace_state_dict


    def load_state_dict(self, state_dict:dict):

        self.data = state_dict['data']

        self.mean = state_dict['mean']
        self.H = state_dict['H']
        self.loss = state_dict['loss']
        self.prior_mean = state_dict['prior_mean']
        self.prior_precision = state_dict['prior_precision']
        self.sigma_noise = state_dict['sigma_noise']
        self.n_data = state_dict['n_data']
        self.n_outputs = state_dict['n_outputs']
        setattr(self.model, 'output_size', self.n_outputs)
        self.likelihood = state_dict['likelihood']
        self.temperature = state_dict['temperature']


    def sample(
        self,
        n_samples: int = 100,
        generator: torch.Generator | None = None,
        covariance_scale_factor: float = 1,
    ) -> torch.Tensor:
        samples = torch.randn(
            n_samples, self.n_params, device=self._device, generator=generator
        )

        if covariance_scale_factor == 1:
            sqrt_gam = 1
        else:
            sqrt_gam = math.sqrt(covariance_scale_factor)

        samples = samples * sqrt_gam*self.posterior_scale.reshape(1, self.n_params)
        return self.mean.reshape(1, self.n_params) + samples
