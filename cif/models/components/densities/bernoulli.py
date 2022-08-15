import torch

from .conditional_density import ConditionalDensity


class BernoulliConditionalDensity(ConditionalDensity):
    def __init__(self, coupler):
        super().__init__()
        self.coupler = coupler

    def _log_prob(self, inputs, cond_inputs):
        all_log_probs = self._get_bernoulli_dist(cond_inputs).log_prob(inputs)
        all_log_probs_flattened = all_log_probs.flatten(start_dim=1)
        log_prob = torch.sum(all_log_probs_flattened, dim=1, keepdim=True)

        return {
            "log-prob": log_prob
        }

    def _sample(self, cond_inputs):
        return {
            "sample": self._get_bernoulli_dist(cond_inputs).sample()
        }

    def _get_bernoulli_dist(self, cond_inputs):
        bernoulli_logits = self.coupler(cond_inputs)
        return torch.distributions.bernoulli.Bernoulli(logits=bernoulli_logits)


def binarize(x):
    x_probs = x / torch.max(x)
    return torch.bernoulli(x_probs)
