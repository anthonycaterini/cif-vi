import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
_DTYPE = torch.get_default_dtype()

from .density import Density
from .gaussian import DiagonalGaussianDensity


class DiagonalGaussianMixtureDensity(Density):
    def __init__(
            self,
            K,
            means,
            log_stddevs,
            log_weights,
            learned_weights=False
    ):
        # XXX: means and log_stddevs are instances of nn.ParameterList and thus get
        #      registered correctly as parameters without being instantiated here
        super().__init__()
        self.K = K
        self.means = means
        self.log_stddevs = log_stddevs
        self.log_weights = nn.Parameter(log_weights, requires_grad=learned_weights)

        self.components = [
            DiagonalGaussianDensity(
                mean=m,
                log_stddev=log_s
            )
            for (m, log_s) in zip(self.means, self.log_stddevs)
        ]

    def _elbo(self, z):
        log_weight_norm = torch.logsumexp(self.log_weights, dim=0)

        prior_log_probs = torch.cat([c.elbo(z)["elbo"] for c in self.components], dim=1)
        joint_log_probs = prior_log_probs + self.log_weights - log_weight_norm
        log_prob = torch.logsumexp(joint_log_probs, dim=1, keepdim=True)

        return {
            "elbo": log_prob
        }

    def _sample(self, num_samples):
        cat_dist = torch.distributions.categorical.Categorical(
            logits=self.log_weights
        )
        sample_inds = cat_dist.sample(sample_shape=(num_samples,))
        sample_mask = F.one_hot(sample_inds, self.K)

        all_samples = torch.stack(
            [c.sample(num_samples) for c in self.components],
            dim=1
        )
        sample_mask_unsqueezed = self._unsqueeze_mask(
            mask=sample_mask,
            count=len(all_samples.shape[2:]),
            start_ind=2
        )
        masked_samples = all_samples*sample_mask_unsqueezed

        return torch.sum(masked_samples, dim=1)

    def _unsqueeze_mask(self, mask, count, start_ind):
        for c in range(count):
            mask = torch.unsqueeze(mask, dim=start_ind+c)
        return mask


# XXX: Below are essentially hard-coded configurations required for factory.py
def get_model_gaussian_mixture_density(layer_config, latent_shape, groundtruth):
    assert layer_config["target"] in ["mog-lattice", "mog-factor"], \
        f"Invalid target {layer_config['target']}"

    def _grad_required(param_key):
        return (
            not groundtruth
            and
            layer_config[f"learnable_{param_key}"]
        )

    # TODO: Make work for higher dimensions
    DIM = 2

    assert latent_shape[0] == DIM, f"Invalid latent shape {latent_shape[0]}"

    K = layer_config["components"]
    sqrt_K = int(np.sqrt(K) + 0.5)

    assert K == sqrt_K ** 2, f"{K} is not a square"

    if _grad_required("mixture_means"):
        means = get_unstructured_mixture_means(latent_shape, K)
    else:
        means = get_structured_mixture_means(layer_config["lim"], sqrt_K)

    if _grad_required("mixture_stddevs"):
        log_stddevs = get_random_mixture_log_stddevs(latent_shape, K)
    else:
        log_stddevs = get_fixed_mixture_log_stddevs(latent_shape, K, layer_config["stddev"])

    if _grad_required("mixture_weights"):
        log_weights = torch.randn(K, dtype=_DTYPE)
    else:
        # TODO: Allow for non-uniform weights
        log_weights = torch.log(torch.ones(K, dtype=_DTYPE)/K)

    return DiagonalGaussianMixtureDensity(
        K=K,
        means=means,
        log_stddevs=log_stddevs,
        log_weights=log_weights,
        learned_weights=_grad_required("mixture_weights")
    )


def get_structured_mixture_means(lim, sqrt_K):
    # TODO: Scale up to higher dimensions
    means = []
    linspace = torch.linspace(-lim, lim, sqrt_K)
    x_coord, y_coord = torch.meshgrid([linspace, linspace])

    for i in range(sqrt_K):
        for j in range(sqrt_K):
            means.append(
                nn.Parameter(
                    torch.tensor(
                        [x_coord[i,j], y_coord[i,j]],
                        dtype=_DTYPE
                    ),
                    requires_grad=False
                )
            )

    return nn.ParameterList(means)


def get_unstructured_mixture_means(latent_shape, K):
    means = []

    for i in range(K):
        means.append(
            nn.Parameter(
                torch.randn(
                    size=latent_shape,
                    dtype=_DTYPE
                ),
                requires_grad=True
            )
        )

    return nn.ParameterList(means)


def get_fixed_mixture_log_stddevs(latent_shape, K, default_stddev):
    single_log_stddev = torch.log(default_stddev*torch.ones(latent_shape, dtype=_DTYPE))
    log_stddevs = nn.ParameterList(
        [nn.Parameter(single_log_stddev, requires_grad=False)]*K
    )

    return nn.ParameterList(log_stddevs)


def get_random_mixture_log_stddevs(latent_shape, K):
    log_stddevs = []

    for i in range(K):
        log_stddev = torch.randn(latent_shape, dtype=_DTYPE)
        log_stddevs.append(nn.Parameter(log_stddev, requires_grad=True))

    return nn.ParameterList(log_stddevs)
