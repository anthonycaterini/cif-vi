from .gaussian import DiagonalGaussianDensity, DiagonalGaussianConditionalDensity
from .diagonal_mog import DiagonalGaussianMixtureDensity, get_model_gaussian_mixture_density

from .vi_density import (
    VIDensity,
    BaseUnconditionalVIPosterior,
    BaseConditionalVIPosterior,
    BijectionVIPosterior,
    CifVIPosterior
)

from .split import SplitDensity
from .vi_targets import BayesianFactorModel, UnconditionalVITarget
from .bernoulli import BernoulliConditionalDensity
