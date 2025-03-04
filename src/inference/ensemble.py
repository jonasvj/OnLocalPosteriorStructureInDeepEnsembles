import torch
import torch.nn as nn
from typing import List, Literal, Tuple
from torch.utils.data import DataLoader
from src.inference import ( InferenceBase, MAP, SWAG, LastLayerLaplace, 
    PosteriorRefinedLastLayerLaplace, SWA, SampleSWAG, IVONFromScratch, MonteCarloDropout)


class DeepEnsemble(InferenceBase):
    """
    Deep Ensemble.
    """
    def __init__(
        self,
        model: List[MAP],
        likelihood: Literal['classification', 'regression'],
    ):
        super().__init__(model=nn.ModuleList(model), likelihood=likelihood)


    def predict(
        self,
        dataloader: DataLoader,
        num_posterior_samples: int = 1,
        stratified: bool = False,
        covariance_scale_factor: float = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts with the deep ensemble.

        Args:
            dataloader: A PyTorch dataloader containing the data to predict.
            num_posterior_samples: Number of posterior samples to compute
                predictions with. This argument is unused as the number of
                posterior samples is equal to the number of ensemble members.
            stratified: Boolean indicating if sampling should be stratified in
                case of a multimodal posterior. This argument is also unused
                as the deep ensemble is stratified by design.

        Returns:
            A tuple of tensors with predictions for the data and the
            corresponding targets. The predictions has shape
            (num_posterior_samples, num_datapoints, output_size) where 
            output_size is the size of a single output from the model and
            num_posterior_samples is equal to the number of ensemble members.
            The targets has shape (num_datapoints,).
        """
        ensemble_outputs = list()
        for model in self.model:
            outputs, targets = model.predict(dataloader)
            ensemble_outputs.append(outputs)

        ensemble_outputs = torch.concat(ensemble_outputs, dim=0)

        return ensemble_outputs, targets


class MultiSWA(DeepEnsemble):
    """
    Multi SWA
    """
    def __init__(
        self,
        model: List[SWA],
        likelihood: Literal['classification', 'regression'],
    ):
        super().__init__(model=model, likelihood=likelihood)


class MultiModalPosterior(InferenceBase):
    """
    Base class that implements the predict function for multimodal posteriors.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def predict(
        self,
        dataloader: DataLoader,
        num_posterior_samples: int = 100,
        stratified: bool = False,
        covariance_scale_factor: float = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts with a multimodal posterior (e.g., MultiSWAG, MoLA, etc.)

        Args:
            dataloader: A PyTorch dataloader containing the data to predict.
            num_posterior_samples: Number of posterior samples to compute
                predictions with.
            stratified: Boolean indicating if sampling should be stratified in
                case of a multimodal posterior.

        Returns:
            A tuple of tensors with predictions for the data and the
            corresponding targets. The predictions has shape
            (num_posterior_samples, num_datapoints, output_size) where 
            output_size is the size of a single output from the model. The 
            targets has shape (num_datapoints,).
        """
        K = len(self.model)

        if stratified:
            multiples = num_posterior_samples // K
            remainders =  num_posterior_samples % K

            # Sample each k equally many times but in random order each time
            sampled_ks = [torch.randperm(K) for _ in range(multiples)]

            # If number of samples is between two multiples of K, distribute
            # remaining samples randomly.
            if remainders > 0:
                sampled_ks.append(torch.multinomial(torch.ones(K)/K, remainders))

            sampled_ks = torch.cat(sampled_ks)            
        else:
            sampled_ks = torch.randint(0, K, (num_posterior_samples,))

        # Unique ks and their counts
        sorted_samples, sort_idx = torch.sort(sampled_ks)
        ks, counts = torch.unique_consecutive(sorted_samples, return_counts=True)

        ensemble_outputs = list()
        for k, count  in zip(ks, counts):
            model = self.model[k]
            outputs, targets = model.predict(
                dataloader,
                num_posterior_samples=count,
                covariance_scale_factor=covariance_scale_factor
            )
            ensemble_outputs.append(outputs)

        ensemble_outputs = torch.concat(ensemble_outputs, dim=0)

        # Retrieve original ordering of samples
        ensemble_outputs = ensemble_outputs[sort_idx.argsort()]

        return ensemble_outputs, targets


class MultiSWAG(MultiModalPosterior):
    """
    MultiSWAG posterior approximation.
    """
    def __init__(
        self,
        model: List[SWAG],
        likelihood: Literal['classification', 'regression'],
    ) -> None:
        super().__init__(model=nn.ModuleList(model), likelihood=likelihood)


class MoLA(MultiModalPosterior):
    """
    Mixture of last-layer Laplace approximations.
    """
    def __init__(
        self,
        model: List[LastLayerLaplace],
        likelihood: Literal['classification', 'regression'],
    ):
        super().__init__(model=nn.ModuleList(model), likelihood=likelihood)


class MoFlowLA(MultiModalPosterior):
    """"
    Mixture of last-layer Laplace approximations refined with normalizing flows.
    """
    def __init__(
        self,
        model: List[PosteriorRefinedLastLayerLaplace],
        likelihood: Literal['classification', 'regression'],
    ):
        super().__init__(model=nn.ModuleList(model), likelihood=likelihood)


class MultiSampleSWAG(MultiModalPosterior):
    """
    MultiSampleSWAG posterior approximation.
    """
    def __init__(
        self,
        model: List[SampleSWAG],
        likelihood: Literal['classification', 'regression'],
    ) -> None:
        super().__init__(model=nn.ModuleList(model), likelihood=likelihood)


class MultiIVONFromScratch(MultiModalPosterior):
    """
    MultiIVONFromScratch posterior approximation.
    """
    def __init__(
        self,
        model: List[IVONFromScratch],
        likelihood: Literal['classification', 'regression'],
    ) -> None:
        super().__init__(model=nn.ModuleList(model), likelihood=likelihood)


class MultiMCDO(MultiModalPosterior):
    """
    MultiMCDO posterior approximation.
    """
    def __init__(
        self,
        model: List[MonteCarloDropout],
        likelihood: Literal['classification', 'regression'],
    ) -> None:
        super().__init__(model=nn.ModuleList(model), likelihood=likelihood)
