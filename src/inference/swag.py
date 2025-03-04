import math
import torch
import torch.nn as nn
from tqdm import trange
from typing import Literal, Tuple
from torch.optim import Optimizer
from src.logging import WandBLogger
from src.inference import InferenceBase
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class SWAG(InferenceBase):
    """
    SWAG posterior of a model.
    """
    def __init__(
        self,
        model: nn.Module,
        likelihood: Literal['classification', 'regression'],
        rank: int = 100,
    ):
        super().__init__(model, likelihood=likelihood)
        self.rank = rank
        self.num_params = sum(p.numel() for p in self.model.parameters())

        self.register_buffer('iterates', torch.zeros(self.num_params, self.rank))
        self.mean = None
        self.diag = None
        self.dev_mat = None


    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        logger: WandBLogger,
    ):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        optimizer = optimizer(params=self.model.parameters())
    
        iterates = list()
        self.model.train()

        step = 0
        pbar = trange(self.rank)
        for epoch in pbar:
            loss_epoch = 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                
                output = self.model(x)
                if self.likelihood == 'classification':
                    loss = self.loss_func(output, y, reduction='sum')
                elif self.likelihood == 'regression':
                    loss = self.loss_func(output, y, reduction='sum', lam=0)

                loss_step = loss / len(y)
                optimizer.zero_grad()
                loss_step.backward()
                optimizer.step()

                # Metrics for tracking
                loss_step = loss_step.detach().item()
                loss_epoch += loss.detach().item()

                # Track step loss
                logger.log('train_loss_step', loss_step, step)

                step += 1
            
            # Save current SGD iterate
            iterates.append(parameters_to_vector(self.model.parameters()).cpu())
            
            # Track epoch loss
            loss_epoch /= len(train_loader.dataset)
            logger.log('train_loss_epoch', loss_epoch, step)
            pbar.set_postfix_str(f'train_loss={loss_epoch:.3e}')

        # Collect SGD iterates in a tensor
        self.iterates = torch.stack(iterates, dim=1)

        # Set model parameters to the posterior mean
        self.compute_objects()
        vector_to_parameters(self.mean.to(device), self.model.parameters())


    def compute_objects(self):
        # Compute objects needed to construct the posterior approximation
        self.mean = self.iterates.mean(dim=1)
        self.diag = torch.clamp((self.iterates**2).mean(dim=1) - self.mean**2, 1e-8)
        self.dev_mat = self.iterates - self.mean.unsqueeze(-1)


    def sample_params(self, covariance_scale_factor: float = 1):
        if self.mean is None or self.diag is None or self.dev_mat is None:
            self.compute_objects()
        
        if covariance_scale_factor == 1:
            sqrt_gam = 1
        else:
            sqrt_gam = math.sqrt(covariance_scale_factor)

        #M, R = self.num_params, self.rank
        M, R = self.iterates.shape
        params_sample = (
            self.mean 
            + sqrt_gam*(1/math.sqrt(2))*torch.sqrt(self.diag)*torch.randn(M)
            + sqrt_gam*(1/math.sqrt(2))*(1/math.sqrt(R - 1))*self.dev_mat @ torch.randn(R)
        )
        return params_sample


    def predict(
        self,
        dataloader: DataLoader,
        num_posterior_samples: int = 100,
        stratified: bool = False,
        covariance_scale_factor: float = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts with the SWAG posterior approximation.

        Args:
            dataloader: A PyTorch dataloader containing the data to predict.
            num_posterior_samples: Number of posterior samples to compute
                predictions with.
            stratified: Boolean indicating if sampling should be stratified in
                case of a multimodal posterior. This argument is unused for the
                SWAG approximation.

        Returns:
            A tuple of tensors with predictions for the data and the
            corresponding targets. The predictions has shape
            (num_posterior_samples, num_datapoints, output_size) where 
            output_size is the size of a single output from the model. The 
            targets has shape (num_datapoints,).
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)

        outputs = list()
        targets = list()

        with torch.no_grad():
            for s in range(num_posterior_samples):
                params_sample = self.sample_params(covariance_scale_factor).to(device)
                vector_to_parameters(params_sample, self.model.parameters())

                outputs_sample = list()
                self.model.eval()
                for x, y in dataloader:
                    x, y = x.to(device), y.to(device)
                    output = self.model(x)

                    outputs_sample.append(output.detach().cpu()) # batch_size x output_size

                    if s == 0:
                        targets.append(y.detach().cpu()) # batch_size

                outputs_sample = torch.concat(outputs_sample, dim=0) # num_datapoints x output_size
                outputs.append(outputs_sample)

        outputs = torch.stack(outputs, dim=0) # num_posterior_samples x num_datapoints x output_size
        targets = torch.concat(targets, dim=0) # num_datapoints

        # Reset model parameters to the posterior mean
        vector_to_parameters(self.mean.to(device), self.model.parameters())

        return outputs, targets


    def load_state_dict(self, state_dict, base: bool = False):
        if not base:
            self.iterates = state_dict.pop('iterates')
        
        for key in list(state_dict.keys()):
            state_dict[key.lstrip('model.')] = state_dict.pop(key)
        self.model.load_state_dict(state_dict)


class SWA(SWAG):
    """
    SWA estimate of a model.
    """

    def compute_objects(self):
        # Compute the SWA solution (mean of SGD iterates)
        self.mean = self.iterates.mean(dim=1)


    def predict(
        self,
        dataloader: DataLoader,
        num_posterior_samples: int = 1,
        stratified: bool = False,
        covariance_scale_factor: float = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts with the SWA solution.

        Args:
            dataloader: A PyTorch dataloader containing the data to predict.
            num_posterior_samples: Number of posterior samples to compute
                predictions with. This argument is unused as the SWA solution
                is used for making predictions.
            stratified: Boolean indicating if sampling should be stratified in
                case of a multimodal posterior. This argument is unused for the
                SWA solution.

        Returns:
            A tuple of tensors with predictions for the data and the
            corresponding targets. The predictions has shape
            (num_posterior_samples, num_datapoints, output_size) where 
            output_size is the size of a single output from the model and
            num_posterior_samples is equal to 1. The targets has shape 
            (num_datapoints,).
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)

        # Insert SWA solution as model parameters
        if self.mean is None:
            self.compute_objects()
        vector_to_parameters(self.mean.to(device), self.model.parameters())

        outputs = list()
        targets = list()

        self.model.eval()
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                output = self.model(x)
                
                outputs.append(output.detach().cpu()) # batch_size x output_size
                targets.append(y.detach().cpu())      # batch_size
        
        outputs = torch.concat(outputs, dim=0) # num_datapoints x output_size
        outputs = outputs.unsqueeze(dim=0) # num_posterior_samples x num_datapoints x output_size
        targets = torch.concat(targets, dim=0) # num_datapoints

        return outputs, targets


class SampleSWAG(SWAG):
    """
    Inference using the samples used to construct SWAG.
    """

    def compute_objects(self):
        # Compute the SWA solution (mean of SGD iterates)
        self.mean = self.iterates.mean(dim=1)


    def predict(
        self,
        dataloader: DataLoader,
        num_posterior_samples: int = 100,
        stratified: bool = False,
        covariance_scale_factor: float = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts with the SWAG samples.

        Args:
            dataloader: A PyTorch dataloader containing the data to predict.
            num_posterior_samples: Number of posterior samples to compute
                predictions with.
            stratified: Boolean indicating if sampling should be stratified in
                case of a multimodal posterior. This argument is unused for the
                SWAG approximation.

        Returns:
            A tuple of tensors with predictions for the data and the
            corresponding targets. The predictions has shape
            (num_posterior_samples, num_datapoints, output_size) where 
            output_size is the size of a single output from the model. The 
            targets has shape (num_datapoints,).
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)

        num_iterates = self.iterates.shape[1]
        if num_posterior_samples > num_iterates:
            num_posterior_samples = num_iterates
            #raise ValueError(
            #    f'This SampleSWAG posterior only has {num_iterates} samples but'
            #    f' {num_posterior_samples} was requested.'
            #)
        sample_indices = torch.multinomial(
            torch.ones(num_iterates) / num_iterates,
            num_samples=num_posterior_samples,
            replacement=False,
        )

        outputs = list()
        targets = list()

        with torch.no_grad():
            for s in range(num_posterior_samples):
                sample_idx = sample_indices[s]
                params_sample = self.iterates[:,sample_idx].to(device)
                vector_to_parameters(params_sample, self.model.parameters())

                outputs_sample = list()
                self.model.eval()
                for x, y in dataloader:
                    x, y = x.to(device), y.to(device)
                    output = self.model(x)

                    outputs_sample.append(output.detach().cpu()) # batch_size x output_size

                    if s == 0:
                        targets.append(y.detach().cpu()) # batch_size

                outputs_sample = torch.concat(outputs_sample, dim=0) # num_datapoints x output_size
                outputs.append(outputs_sample)

        outputs = torch.stack(outputs, dim=0) # num_posterior_samples x num_datapoints x output_size
        targets = torch.concat(targets, dim=0) # num_datapoints

        # Reset model parameters to the posterior mean
        if self.mean is None:
            self.compute_objects()
        vector_to_parameters(self.mean.to(device), self.model.parameters())

        return outputs, targets