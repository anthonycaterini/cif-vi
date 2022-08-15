import tqdm

import numpy as np

import torch
import torch.nn as nn

from .density import Density
from .bernoulli import binarize


class VIDensity(Density):
    def __init__(
            self,
            vi_posterior,
            target,
            groundtruth_target=None
    ):
        super().__init__()
        self.vi_posterior = vi_posterior
        self.target = target
        self.groundtruth_target = groundtruth_target

    def forward(self, mode, *args, **kwargs):
        if mode == "groundtruth-sample":
            return self._groundtruth_sample(*args)
        elif mode == "marginal-elbo":
            return self._marginal_elbo(*args)
        elif mode == "approx-posterior-sample":
            return self._approx_posterior_sample(*args)
        else:
            return super().forward(mode, *args, **kwargs)

    def groundtruth_sample(self, num_samples):
        return self("groundtruth-sample", num_samples)

    def marginal_elbo(self, x, z, num_elbo_samples):
        return self("marginal-elbo", x, z, num_elbo_samples)

    def approx_posterior_sample(self, x, num_samples):
        return self("approx-posterior-sample", x, num_samples)

    def _elbo(self, x, **kwargs):
        if len(x.shape) == 4:
            x = binarize(x)

        result = self.vi_posterior.elbo(x, **kwargs)
        approx_posterior_sample = result["sample"]

        elbo = self.target.elbo(x, approx_posterior_sample)["elbo"] - result["elbo"]
        return {
            "elbo": elbo,
            "approx-posterior-sample": approx_posterior_sample
        }

    def _marginal_elbo(self, x, z, num_elbo_samples):
        log_target = self.target.elbo(x, z)["elbo"]

        z_repeated = z.repeat_interleave(num_elbo_samples, dim=0)
        with torch.no_grad():
            posterior_result = self.vi_posterior.posterior_elbo(z_repeated)["posterior-elbo"]

        posterior_elbo_samples = posterior_result.view(z.shape[0], num_elbo_samples)
        posterior_estimate = posterior_elbo_samples.logsumexp(dim=1, keepdim=True) - np.log(num_elbo_samples)

        return {
            "marginal-elbo": log_target - posterior_estimate,
        }

    def _sample(self, num_samples):
        return self.target.sample(num_samples)

    def _approx_posterior_sample(self, x, num_samples):
        return {
            "approx-posterior-sample": self.vi_posterior.elbo(x, num_samples=num_samples)["sample"]
        }

    def _groundtruth_sample(self, num_samples):
        assert self.groundtruth_target, "No groundtruth target"
        return self.groundtruth_target.sample(num_samples)


class VIPosterior(Density):
    def forward(self, mode, *args, **kwargs):
        if mode == "elbo":
            return self._elbo(*args, **kwargs)

        elif mode == "posterior-elbo":
            return self._posterior_elbo(*args)

        else:
            assert False, f"Invalid mode {mode}"

    def posterior_elbo(self, z):
        return self._posterior_elbo(z)

    def _posterior_elbo(self, z):
        raise NotImplementedError


class BaseUnconditionalVIPosterior(VIPosterior):
    def __init__(
            self,
            prior_density
    ):
        # prior_density is an object of subclass Density
        super().__init__()
        self.prior_densiy = prior_density

    def _elbo(self, x, num_samples=None):
        if num_samples:
            prior_sample = self.prior_densiy.sample(num_samples)
        else:
            prior_sample = self.prior_densiy.sample(x.shape[0])
        prior_elbo = self.prior_densiy.elbo(prior_sample)

        return {
            "elbo": prior_elbo["elbo"],
            "sample": prior_sample
        }

    def _posterior_elbo(self, z):
        return {
            "posterior-elbo": self.prior_densiy.elbo(z)["elbo"]
        }


class BaseConditionalVIPosterior(VIPosterior):
    def __init__(
            self,
            prior_density
    ):
        # prior_density is an object of subclass ConditionalDensity
        super().__init__()
        self.prior_density = prior_density

    def _elbo(self, x):
        prior_result = self.prior_density.sample(x)
        return {
            "elbo": prior_result["log-prob"],
            "sample": prior_result["sample"]
        }


class BijectionVIPosterior(VIPosterior):
    def __init__(
            self,
            vi_prior,
            bijection
    ):
        super().__init__()
        self.vi_prior = vi_prior
        self.bijection = bijection

    def _elbo(self, x, **kwargs):
        vi_prior_result = self.vi_prior.elbo(x, **kwargs)
        bijection_result = self.bijection.x_to_z(vi_prior_result["sample"])

        return {
            "elbo": vi_prior_result["elbo"] - bijection_result["log-jac"],
            "sample": bijection_result["z"]
        }

    def _posterior_elbo(self, z):
        bijection_result = self.bijection.z_to_x(z)
        vi_prior_result = self.vi_prior.posterior_elbo(bijection_result["x"])

        return {
            "posterior-elbo": vi_prior_result["posterior-elbo"] + bijection_result["log-jac"]
        }


class CifVIPosterior(VIPosterior):
    def __init__(
            self,
            vi_prior,
            bijection,
            q_u_given_w,
            r_u_given_z,
            amortized
    ):
        super().__init__()
        self.vi_prior = vi_prior
        self.bijection = bijection
        self.q_u_given_w = q_u_given_w
        self.r_u_given_z = r_u_given_z
        self.amortized = amortized

    def _elbo(self, x, **kwargs):
        vi_prior_result = self.vi_prior.elbo(x, **kwargs)
        w_prev = vi_prior_result["sample"]
        log_prior = vi_prior_result["elbo"]

        q_u_result = self.q_u_given_w.sample(w_prev)
        log_q_u = q_u_result["log-prob"]
        u = q_u_result["sample"]

        bijection_result = self.bijection.x_to_z(w_prev, u=u)
        w = bijection_result["z"]
        log_jac = bijection_result["log-jac"]

        r_cond_input = (w, x) if self.amortized else w
        log_r_u = self.r_u_given_z.log_prob(u, r_cond_input)["log-prob"]

        return {
            "elbo": log_prior + log_q_u - log_jac - log_r_u,
            "sample": w,
            "u": u
        }

    def _posterior_elbo(self, z):
        assert not self.amortized, "Amortized posterior ELBO not yet implemented"

        r_u_result = self.r_u_given_z.sample(z)
        log_r_u = r_u_result["log-prob"]
        u = r_u_result["sample"]

        bijection_result = self.bijection.z_to_x(z, u=u)
        w = bijection_result["x"]
        log_jac = bijection_result["log-jac"]

        log_q_u = self.q_u_given_w.log_prob(u, w)["log-prob"]
        vi_prior_result = self.vi_prior.posterior_elbo(w)["posterior-elbo"]

        return {
            "posterior-elbo": vi_prior_result + log_q_u + log_jac - log_r_u,
            "u": u
        }
