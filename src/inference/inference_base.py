import torch
import torch.nn as nn
import uncertainty_toolbox as uct
import torch.distributions as dist
from ood_metrics import calc_metrics
from torch.utils.data import DataLoader
from typing import Literal, Union, List, Dict, Tuple, Callable
from torchmetrics.functional.classification import multiclass_calibration_error
from src.utils import interpolation_loss, regression_ece_for_gaussian_mixture, ence

all_regression_metrics = [
    'avg_total_unc', 'avg_aleatoric_unc', 'avg_epistemic_unc', 'lpd',
    'z_score_var', 'ence', 'rmv', 'cv', 'ece', 'mae', 'rmse', 'mdae', 'marpd',
    'r2', 'corr', 'rms_cal', 'ma_cal', 'miscal_area', 'sharp', 'nll', 'crps',
    'check', 'interval', 'loss'
]

class InferenceBase(nn.Module):
    """
    Base inference class that implements methods that are common to all 
    inference classes.
    """
    def __init__(
        self,
        model: Union[nn.Module, nn.ModuleList],
        likelihood: Literal['classification', 'regression'],
    ) -> None:
        super().__init__()
        self.model = model
        self.likelihood = likelihood
        self.lam = 1. # Used for interpolation loss in regression metrics
        self.y_var = 1. # Used for interpolation loss in regression metrics


    @property
    def loss_func(self) -> Callable:
        """
        Loss function to fit the model with.
        """
        if self.likelihood == 'classification':
            return nn.functional.cross_entropy
        elif self.likelihood == 'regression':
            return interpolation_loss


    def fit(self) -> None:
        """
        Fits model.
        """
        raise NotImplementedError


    def predict(
        self,
        dataloader: DataLoader,
        num_posterior_samples: int = 100,
        stratified: bool = False,
        covariance_scale_factor: float = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts with the model.

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
        raise NotImplementedError


    def compute_stats(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        subsample_sizes: Union[int, List[int], None] = None
    ) -> List[Dict]:
        """
        Compute evaluation statistics depending on the likelihood type.

        Args:
            outputs: A tensor with model outputs that has shape 
                (num_posterior_samples, num_datapoints, output_size) where
                output_size is the size of a single output from the model.
            targets: A tensor with prediction targets that has shape
                (num_datapoints,).
            subsample_sizes: Integer or list of integers that indicates the 
                number of posterior samples to use when computing statistics. 
                Defaults to None in which case all posterior samples are used.

        Returns:
            A list of dictionaries with evaluation statistics for each
            subsample size.
        """
        assert outputs.shape[1] == targets.shape[0] # Same number of datapoints

        if subsample_sizes is None:
            subsample_sizes = [len(outputs)]
        elif isinstance(subsample_sizes, int):
            subsample_sizes = [subsample_sizes]

        subsample_sizes = sorted(subsample_sizes)

        # Subsamples must be smaller than or equal to the full sample
        #assert subsample_sizes[-1] <= len(outputs) 
        if subsample_sizes[-1] > len(outputs):
            subsample_sizes = [s for s in subsample_sizes if s < len(outputs)]
            subsample_sizes.append(len(outputs))

        # Compute evaluation statistics using an increasing number of posterior
        # samples
        stats_by_sample_size = list()
        for num_samples in subsample_sizes:
            # The stratification in self.predict returns the outputs in an 
            # order that ensures that this is still stratified
            outputs_subset = outputs[:num_samples]

            if self.likelihood == 'classification':
                stats = self.compute_classification_stats(outputs_subset, targets)
            elif self.likelihood == 'regression':
                stats = self.compute_regression_stats(outputs_subset, targets)

            stats['num_posterior_samples'] = num_samples
            stats_by_sample_size.append(stats)

        return stats_by_sample_size


    def compute_ood_stats(
        self,
        outputs_id: torch.Tensor,
        outputs_ood: torch.Tensor,
        subsample_sizes: Union[int, List[int], None] = None
    ) -> List[Dict]:
        """
        Compute out-of-distribution statistics depending on the likelihood type.

        Args:
            outputs_id: A tensor with model outputs arising from 
                in-distribution data that has shape (num_posterior_samples,
                num_id_datapoints, output_size) where output_size is the size 
                of a single output from the model.
            outputs_ood: A tensor with model outputs arising from 
                out-of-distribution data that has shape (num_posterior_samples,
                num_ood_datapoints, output_size) where output_size is the size 
                of a single output from the model.
            subsample_sizes: Integer or list of integers that indicates the 
                number of posterior samples to use when computing statistics. 
                Defaults to None in which case all posterior samples are used.

        Returns:
            A list of dictionaries with out-of-distribution statistics for each
            subsample size.
        """
        assert outputs_id.shape[0] == outputs_ood.shape[0] # Same number of posterior samples
        assert outputs_id.shape[2] == outputs_ood.shape[2] # Same number of model outputs

        if subsample_sizes is None:
            subsample_sizes = [len(outputs_id)]
        elif isinstance(subsample_sizes, int):
            subsample_sizes = [subsample_sizes]

        subsample_sizes = sorted(subsample_sizes)

        # Subsamples must be smaller than or equal to the full sample
        #assert subsample_sizes[-1] <= len(outputs_id)
        if subsample_sizes[-1] > len(outputs_id):
            subsample_sizes = [s for s in subsample_sizes if s < len(outputs_id)]
            subsample_sizes.append(len(outputs_id))

        # Compute out-of-distribution statistics using an increasing number of
        # posterior samples
        stats_by_sample_size = list()
        for num_samples in subsample_sizes:
            # The stratification in self.predict returns the outputs in an 
            # order that ensures that this is still stratified
            outputs_id_subset = outputs_id[:num_samples]
            outputs_ood_subset = outputs_ood[:num_samples]

            if self.likelihood == 'classification':
                ood_stats = self.compute_classification_ood_stats(
                    outputs_id_subset, outputs_ood_subset
                )
            elif self.likelihood == 'regression':
                ood_stats = self.compute_regression_ood_stats(
                    outputs_id_subset, outputs_ood_subset
                )

            ood_stats['num_posterior_samples'] = num_samples
            stats_by_sample_size.append(ood_stats)

        return stats_by_sample_size


    def compute_classification_stats(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict:
        """
        Compute evaluation statistics for classification.

        Args:
            logits: A tensor with shape (num_posterior_samples, num_datapoints,
                num_classes).
            targets: A tensor with shape (num_datapoints,) with prediction
                targets.
        Returns:
            A dictionary with the evaluation statistics.
        """
        eval_stats = dict()

        S, N, C = logits.shape

        # Probabilities
        probs = torch.softmax(logits, dim=-1)                       # S x N x C
        avg_probs = torch.mean(probs, dim=0)                        #     N x C

        # Predicted class
        class_preds = torch.argmax(avg_probs, dim=-1)               #     N

        # Loss
        # (Note: NLL loss expects log probabilities and not logits like CE loss does.)
        eval_stats['loss'] = nn.functional.nll_loss(
            torch.log(avg_probs), targets, reduction='mean'
        ).item()

        # LPD
        log_densities = dist.Categorical(logits=logits).log_prob(targets) # S x N
        eval_stats['lpd'] = (
            -N*torch.log(torch.tensor(S))
            + torch.logsumexp(log_densities, dim=0).sum() 
        ).item() / N

        # Accuracy
        eval_stats['acc'] = torch.sum(
            class_preds == targets
        ).item() / N

        # Average confidence
        eval_stats['avg_conf'] = torch.sum(
            torch.max(avg_probs, dim=-1).values
        ).item() / N

        # Average entropy
        entropy = -torch.sum(
            torch.where(avg_probs == 0., 0., avg_probs*torch.log(avg_probs)),
            dim=-1
        )
        eval_stats['avg_entropy'] = torch.sum(entropy).item() / N

        # Calibration error
        eval_stats['ece'] = multiclass_calibration_error(
            avg_probs, targets, C, norm='l1'
        ).item()
        eval_stats['mce'] = multiclass_calibration_error(
            avg_probs, targets, C, norm='max'
        ).item()

        # Brier score
        eval_stats['brier'] = (
            (avg_probs - nn.functional.one_hot(targets))**2
        ).sum().item() / N

        return eval_stats


    def compute_classification_ood_stats(
        self,
        logits_id: torch.Tensor,
        logits_ood: torch.Tensor,
    ) -> Dict:
        """
        Compute out-of-distribution statistics for classification.

        Args:
            logits_id: A tensor with shape (num_posterior_samples,
                num_id_datapoints, num_classes) of logits arising from
                predictions on in-distribution data.
            logits_ood: A tensor with shape (num_posterior_samples,
                num_ood_datapoints, num_classes) of logits arising from
                predictions on out-of-distribution data.
        Returns:
            A dictionary with the out-of-distribution statistics.
        """
        eval_stats = dict()

        # Probabilities
        probs_id = torch.softmax(logits_id, dim=-1)                       # S x N x C
        avg_probs_id = torch.mean(probs_id, dim=0)                        #     N x C
        probs_ood = torch.softmax(logits_ood, dim=-1)
        avg_probs_ood = torch.mean(probs_ood, dim=0)

        # Confidences
        conf_id = torch.max(avg_probs_id, dim=-1).values
        conf_ood = torch.max(avg_probs_ood, dim=-1).values

        # Entropies
        entropy_id = -torch.sum(
            torch.where(avg_probs_id == 0., 0., avg_probs_id*torch.log(avg_probs_id)),
            dim=-1
        )
        entropy_ood = -torch.sum(
            torch.where(avg_probs_ood == 0., 0., avg_probs_ood*torch.log(avg_probs_ood)),
            dim=-1
        )

        # Average confidence
        eval_stats['avg_conf_id'] = torch.mean(conf_id).item()
        eval_stats['avg_conf_ood'] = torch.mean(conf_ood).item()

        # Average entropy
        eval_stats['avg_entropy_id'] = torch.mean(entropy_id).item()
        eval_stats['avg_entropy_ood'] = torch.mean(entropy_ood).item()

        # Out-of-distribution detection (using the entropy as a predictor)
        scores = torch.concat([entropy_id, entropy_ood])
        labels = torch.concat([torch.zeros_like(entropy_id), torch.ones_like(entropy_ood)])

        ood_metrics = calc_metrics(scores.numpy(), labels.numpy())
        eval_stats.update(ood_metrics)

        return eval_stats


    def compute_regression_stats(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict:
        """
        Compute evaluation statistics for regression.

        Args:
            outputs: A tensor with shape (num_posterior_samples, num_datapoints,
            output_size) where output_size is the size of a single output from 
            the model.
            targets: A tensor with shape (num_datapoints,) with prediction
                targets.
        Returns:
            A dictionary with the evaluation statistics.
        """
        eval_stats = dict()
        outputs = outputs.detach().to('cpu')                            # S x N x K
        targets = targets.detach().to('cpu').squeeze()                  #     N
        S, N, K = outputs.shape

        # Assume heteroschedastic regression
        assert K == 2
        model_means = outputs[:,:,0]                                    # S x N
        model_vars = outputs[:,:,1]                                     # S x N
        ensemble_mean = model_means.mean(dim=0)                         #     N
        ensemble_aleatoric_var = model_vars.mean(dim=0)                 #     N
        #ensemble_epistemic_var = (model_means**2).mean(dim=0) - ensemble_mean**2# N
        ensemble_epistemic_var = torch.mean((model_means - ensemble_mean)**2, dim=0)
        ensemble_total_var = ensemble_aleatoric_var + ensemble_epistemic_var    # N

        # Average uncertainties
        eval_stats['avg_total_unc'] = ensemble_total_var.mean().item()
        eval_stats['avg_aleatoric_unc'] = ensemble_aleatoric_var.mean().item()
        eval_stats['avg_epistemic_unc'] = ensemble_epistemic_var.mean().item()

        # Expected log predictive density
        log_densities = dist.Normal(
            model_means, torch.sqrt(model_vars)
        ).log_prob(targets)                                             # S x N
        eval_stats['lpd'] = (
            -N*torch.log(torch.tensor(S))
            + torch.logsumexp(log_densities, dim=0).sum() 
        ).item() / N

        # Z-score variance
        ensemble_std = torch.sqrt(ensemble_total_var)
        eval_stats['z_score_var'] = (
            (targets - ensemble_mean) / ensemble_std
        ).var().item()

        # Expected normalized calibration error
        eval_stats['ence'] = ence(ensemble_mean, ensemble_total_var, targets).item()

        # Root mean variance
        eval_stats['rmv'] = torch.sqrt(ensemble_total_var.mean()).item()

        # Coefficient of variation
        mean_std = ensemble_std.mean()
        eval_stats['cv'] = (
            torch.sqrt(torch.mean((ensemble_std - mean_std)**2)) / mean_std
        ).item()

        # Expected calibration error / Mean absolute calibration error
        eval_stats['ece'] = regression_ece_for_gaussian_mixture(
            model_means.numpy(),
            torch.sqrt(model_vars).numpy(),
            targets.numpy(),
        )

        # Uncertainty toolbox metrics
        uct_metrics = uct.get_all_metrics(
            ensemble_mean.numpy(),
            torch.sqrt(ensemble_total_var).numpy(),
            targets.numpy(),
            verbose=False,
        )
        for key, d in uct_metrics.items():
            if key != 'adv_group_calibration':
                eval_stats.update(d)

        # Loss (used for monitoring convergence)
        eval_stats['loss'] = interpolation_loss(
            torch.stack([ensemble_mean, ensemble_total_var], dim=1),
            targets,
            reduction='mean',
            lam=self.lam,
            y_var=self.y_var,
        ).item()

        return eval_stats


    def compute_regression_ood_stats(
        self,
        outputs_id: torch.Tensor,
        outputs_ood: torch.Tensor,
    ) -> Dict:
        """
        Compute out-of-distribution statistics for regression.

        Args:
            outputs_id: A tensor with shape (num_posterior_samples,
                num_id_datapoints, output_size) of model outputs arising from
                predictions on in-distribution data.
            outputs_ood: A tensor with shape (num_posterior_samples,
                num_ood_datapoints, output_size) of model outputs arising from
                predictions on out-of-distribution data.
        Returns:
            A dictionary with the out-of-distribution statistics.
        """
        eval_stats = dict()
        outputs_id = outputs_id.detach().to('cpu')                              # S x N x K
        outputs_ood = outputs_ood.detach().to('cpu')                            # S x N x K
        S_id, N_id, K_id = outputs_id.shape
        S_ood, N_ood, K_ood = outputs_ood.shape

        # Assume heteoschedastic regression
        assert K_id == 2 and K_ood == 2

        # ID predictions
        model_means_id = outputs_id[:,:,0]                                      # S x N
        model_vars_id = outputs_id[:,:,1]                                       # S x N
        ensemble_mean_id = model_means_id.mean(dim=0)                           #     N
        ensemble_aleatoric_var_id = model_vars_id.mean(dim=0)                   #     N
        ensemble_epistemic_var_id = torch.mean((model_means_id - ensemble_mean_id)**2, dim=0) # N
        ensemble_total_var_id = ensemble_aleatoric_var_id + ensemble_epistemic_var_id # N

        eval_stats['avg_total_unc_id'] = ensemble_total_var_id.mean().item()
        eval_stats['avg_aleatoric_unc_id'] = ensemble_aleatoric_var_id.mean().item()
        eval_stats['avg_epistemic_unc_id'] = ensemble_epistemic_var_id.mean().item()

        # OOD predictions
        model_means_ood = outputs_ood[:,:,0]                                    # S x N
        model_vars_ood = outputs_ood[:,:,1]                                     # S x N
        ensemble_mean_ood = model_means_ood.mean(dim=0)                         #     N
        ensemble_aleatoric_var_ood = model_vars_ood.mean(dim=0)                 #     N
        ensemble_epistemic_var_ood = torch.mean((model_means_ood - ensemble_mean_ood)**2, dim=0) # N
        ensemble_total_var_ood = ensemble_aleatoric_var_ood + ensemble_epistemic_var_ood # N

        eval_stats['avg_total_unc_ood'] = ensemble_total_var_ood.mean().item()
        eval_stats['avg_aleatoric_unc_ood'] = ensemble_aleatoric_var_ood.mean().item()
        eval_stats['avg_epistemic_unc_ood'] = ensemble_epistemic_var_ood.mean().item()

        # Out-of-distribution detection (using the total variance as a predictor)
        scores = torch.concat([ensemble_total_var_id, ensemble_total_var_ood])
        labels = torch.concat([torch.zeros_like(ensemble_total_var_id), torch.ones_like(ensemble_total_var_ood)])

        ood_metrics = calc_metrics(scores.numpy(), labels.numpy())
        eval_stats.update(ood_metrics)

        return eval_stats
