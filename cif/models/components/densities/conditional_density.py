import torch.nn as nn


class ConditionalDensity(nn.Module):
    def forward(self, mode, *args):
        if mode == "log-prob":
            return self._log_prob(*args)
        elif mode == "sample":
            return self._sample(*args)
        else:
            assert False, f"Invalid mode {mode}"

    def log_prob(self, inputs, cond_inputs):
        return self("log-prob", inputs, cond_inputs)

    def sample(self, cond_inputs):
        return self("sample", cond_inputs)

    def _log_prob(self, inputs, cond_inputs):
        raise NotImplementedError

    def _sample(self, cond_inputs):
        raise NotImplementedError
