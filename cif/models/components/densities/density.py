import numpy as np
import torch.nn as nn


class Density(nn.Module):
    def forward(self, mode, *args, **kwargs):
        if mode == "elbo":
            return self._elbo(*args, **kwargs)

        elif mode == "sample":
            return self._sample(*args)

        elif mode == "fixed-sample":
            return self._fixed_sample(*args)

        else:
            assert False, f"Invalid mode {mode}"

    def fix_random_u(self):
        fixed_density, _ = self._fix_random_u()
        return fixed_density

    def fix_u(self, u):
        raise NotImplementedError

    def elbo(self, x, **kwargs):
        return self("elbo", x, **kwargs)

    def sample(self, num_samples):
        return self("sample", num_samples)

    def fixed_sample(self, noise=None):
        return self("fixed-sample", noise)

    def _fix_random_u(self):
        raise NotImplementedError

    def _elbo(self, x):
        raise NotImplementedError

    def _sample(self, num_samples):
        raise NotImplementedError

    def _fixed_sample(self, noise):
        raise NotImplementedError
