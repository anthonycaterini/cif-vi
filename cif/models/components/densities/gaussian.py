import numpy as np
import torch
import torch.nn as nn

from .conditional_density import ConditionalDensity
from .density import Density


def diagonal_gaussian_log_prob(w, means, log_stddevs):
    assert means.shape == log_stddevs.shape == w.shape

    flat_w = w.flatten(start_dim=1)
    flat_means = means.flatten(start_dim=1)
    flat_log_stddevs = log_stddevs.flatten(start_dim=1)
    flat_vars = torch.exp(flat_log_stddevs)**2

    _, dim = flat_w.shape

    const_term = -.5*dim*np.log(2*np.pi)
    log_det_terms = -torch.sum(flat_log_stddevs, dim=1, keepdim=True)
    product_terms = -.5*torch.sum((flat_w - flat_means)**2 / flat_vars, dim=1, keepdim=True)

    return const_term + log_det_terms + product_terms


def diagonal_gaussian_sample(means, log_stddevs):
    epsilon = torch.randn_like(means)
    samples = torch.exp(log_stddevs)*epsilon + means

    log_probs = diagonal_gaussian_log_prob(samples, means, log_stddevs)

    return samples, log_probs


def diagonal_gaussian_entropy(stddevs):
    flat_stddevs = stddevs.flatten(start_dim=1)
    _, dim = flat_stddevs.shape
    return torch.sum(torch.log(flat_stddevs), dim=1, keepdim=True) + .5*dim*(1 + np.log(2*np.pi))


class DiagonalGaussianDensity(Density):
    def __init__(self, mean, log_stddev, num_fixed_samples=0):
        super().__init__()
        assert mean.shape == log_stddev.shape

        if mean.requires_grad:
            if type(mean).__name__ == "Tensor":
                self.mean = nn.Parameter(mean)
            else:
                # XXX: This case happen when `mean` is an element of a nn.ParameterList,
                #      meaning it is already properly registered
                self.mean = mean
        else:
            self.register_buffer("mean", mean)

        if log_stddev.requires_grad:
            if type(log_stddev).__name__ == "Tensor":
                self.log_stddev = nn.Parameter(log_stddev)
            else:
                # XXX: Same case as above with `self.mean`
                self.log_stddev = log_stddev
        else:
            self.register_buffer("log_stddev", log_stddev)

        if num_fixed_samples > 0:
            self.register_buffer("_fixed_samples", self.sample(num_fixed_samples))

    @property
    def shape(self):
        return self.mean.shape

    def _fix_random_u(self):
        return self, self.sample(num_samples=1)[0]

    def fix_u(self, u):
        assert not u
        return self

    def _elbo(self, z):
        log_prob = diagonal_gaussian_log_prob(
            z,
            self.mean.expand_as(z),
            self.log_stddev.expand_as(z),
        )
        return {
            "elbo": log_prob,
            "z": z
        }

    def _sample(self, num_samples):
        samples, _ = diagonal_gaussian_sample(
            self.mean.expand(num_samples, *self.shape),
            self.log_stddev.expand(num_samples, *self.shape)
        )
        return samples

    def _fixed_sample(self, noise):
        return noise if noise is not None else self._fixed_samples


class DiagonalGaussianConditionalDensity(ConditionalDensity):
    def __init__(
            self,
            coupler
    ):
        super().__init__()
        self.coupler = coupler

    def _log_prob(self, inputs, cond_inputs):
        means, log_stddevs = self._means_and_log_stddevs(cond_inputs)
        return {
            "log-prob": diagonal_gaussian_log_prob(inputs, means, log_stddevs)
        }

    def _sample(self, cond_inputs):
        means, log_stddevs = self._means_and_log_stddevs(cond_inputs)
        samples, log_probs = diagonal_gaussian_sample(means, log_stddevs)
        return {
            "log-prob": log_probs,
            "sample": samples
        }

    def _means_and_log_stddevs(self, cond_inputs):
        result = self.coupler(cond_inputs)
        return result["shift"], result["log-scale"]
