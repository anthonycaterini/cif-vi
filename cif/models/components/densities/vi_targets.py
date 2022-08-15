from .density import Density


class BaseVITarget(Density):
    def elbo(self, data, latent):
        return self("elbo", data, latent)


class BayesianFactorModel(BaseVITarget):
    def __init__(
            self,
            prior,
            likelihood
    ):
        # prior is an object of subclass Density
        # likelihood is an object of subclass ConditionalDensity
        super().__init__()
        self.prior = prior
        self.likelihood = likelihood

    def forward(self, mode, *args):
        if mode == "sample-from-latent":
            return self._sample_from_latent(*args)
        else:
            return super().forward(mode, *args)

    def sample_from_latent(self, latent):
        return self("sample-from-latent", latent)

    def _elbo(self, data, latent):
        prior_log_prob = self.prior.elbo(latent)["elbo"]
        likelihood_log_prob = self.likelihood.log_prob(data, latent)["log-prob"]
        return {
            "elbo": prior_log_prob + likelihood_log_prob
        }

    def _sample(self, num_samples):
        """ Sample directly from the generative model """
        latents = self.prior.sample(num_samples)
        return self.sample_from_latent(latents)

    def _sample_from_latent(self, latent):
        return self.likelihood.sample(latent)["sample"]


class UnconditionalVITarget(BaseVITarget):
    def __init__(
            self,
            posterior_density
    ):
        # posterior_density is an object of subclass Density
        super().__init__()
        self.posterior_density = posterior_density

    def _elbo(self, data, latent):
        return self.posterior_density.elbo(latent)
