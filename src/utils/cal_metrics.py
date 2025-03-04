import math
import torch
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.optimize import root_scalar


C = np.sqrt(2*np.pi)

def standard_gaussian_pdf(x):
    return np.exp(-(x**2)/2.0)/C


class GaussianMixtureDistributionNumpy:
    def __init__(self, means: np.array, scales: np.array, weights: np.array):
        self.means = means
        self.scales = scales
        self.weights = weights
        self.K = len(means)

    """
    def objective(self, x, p):
        x_tilde =  (x - self.means) / self.scales
        return (
            np.inner(self.weights, ndtr(x_tilde)) - p, # CDF - p
            np.inner(self.weights, standard_gaussian_pdf(x_tilde) / self.scales), # PDF (derivative of CDF - p)
        )
    """

    def objective(self, x, p):
        return np.inner(self.weights, ndtr((x - self.means) / self.scales)) - p # CDF - p


    def optim_range(self, p):
        quantiles = ndtri(p)*self.scales + self.means
        a, b = np.min(quantiles), np.max(quantiles)
        return (a + b) / 2, [a, b]


    def ppf(self, p):
        if self.K == 1:
            return ndtri(p)*self.scales + self.means

        x0, bracket = self.optim_range(p)
        obj_func = lambda x: self.objective(x, p)
        try:
            r = root_scalar(
                f=obj_func,
                fprime=False,
                fprime2=False,
                bracket=bracket,
                x0=x0,
            )
            assert r.converged
            return r.root
        except ValueError as e:
            #print('Finding a solution to CDF(x) - p = 0 failed. Reason:')
            #print(e)
            #print('Returning value in the set {a, (a+b)/2, b} closest to a solution instead.')
            x_vals = [bracket[0], x0, bracket[1]]
            obj_func_vals = np.array([obj_func(bracket[0]), obj_func(x0), obj_func(bracket[1])])
            idx_min = np.argmin(np.abs(obj_func_vals))
            #print('[a, (a+b)/2, b]:', x_vals)
            #print('f([a, (a+b)/2, b]):', obj_func_vals)
            #print('idx_min of abs(f([a, (a+b)/2, b])):', idx_min)
            #print('x_vals[idx_min]:', x_vals[idx_min])   
            return x_vals[idx_min]



def regression_ece_for_gaussian_mixture(
    means: np.ndarray,
    scales: np.ndarray,
    targets: np.ndarray,
    probs: np.ndarray = np.linspace(0.01, 0.99, 20),
) -> float:
    """
    Computes the regression ECE for a gaussian mixture predictive distribution
    with uniform weights.

    Args:
        means: np.ndarray of shape S x N of predictive means.
        scales: np.ndarray of shape S x N of predictive standard deviations.
        targets: np.ndarray of shape (N,) with prediction targets.
        probs: np.ndarray with probabilities to compute percentiles for.

    Returns:
        The regression ECE as a float.
    """
    S, N = means.shape
    weights = np.ones(S)/S

    # Compute observed probabilities
    observed_probs = np.zeros_like(probs)
    for n in range(N):
        target = targets[n]
        m, s = means[:,n], scales[:,n]
        gmm = GaussianMixtureDistributionNumpy(means=m, scales=s, weights=weights)

        for prob_idx, prob in enumerate(probs):
            percentile = gmm.ppf(prob)
            if target <= percentile:
                observed_probs[prob_idx:] += 1
                break

    observed_probs /= N
    ece = np.abs(observed_probs - probs).mean()

    return ece


def ence(
    ensemble_mean: torch.Tensor,
    ensemble_var: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int = 15
) -> torch.Tensor:
    """
    Computes expected normalized calibration error.

    Args:
        ensemble_mean: torch.Tensor with shape (N,) with the predictive means.
        ensemble_var: torch.Tensor with shape (N,) with the predictive
            variances.
        targets: torch.Tensor with shape (N,) with the targets values.
        num_bins: Number of bins (int) to use for the ENCE calculation.
    
    Returns
        A 0-dimensional tensor (float) with the computed ENCE.
    """
    sort_idx = torch.argsort(torch.sqrt(ensemble_var))
    bin_idx = torch.split(sort_idx, math.ceil(len(sort_idx) / num_bins))
    assert len(bin_idx) == num_bins

    rmv = torch.empty(num_bins)
    rmse = torch.empty(num_bins)
    squared_errors = (ensemble_mean - targets)**2
    for j, bin in enumerate(bin_idx):
        rmv[j] = torch.sqrt(ensemble_var[bin].mean())
        rmse[j] = torch.sqrt(squared_errors[bin].mean())
    
    ence = (torch.abs(rmv - rmse) / rmv).mean()
    return ence