import numpy as np
import torch
import torch.nn as nn


from .components.bijections import (
    FlipBijection,
    MADEBijection,
    BatchNormBijection,
    ViewBijection,
    ConditionalAffineBijection,
    CoupledRationalQuadraticSplineBijection,
    AutoregressiveRationalQuadraticSplineBijection,
    LULinearBijection,
    RandomChannelwisePermutationBijection
)
from .components.densities import (
    DiagonalGaussianDensity,
    DiagonalGaussianConditionalDensity,
    get_model_gaussian_mixture_density,
    VIDensity,
    BaseConditionalVIPosterior,
    BaseUnconditionalVIPosterior,
    BijectionVIPosterior,
    CifVIPosterior,
    BayesianFactorModel,
    UnconditionalVITarget,
    BernoulliConditionalDensity,
    SplitDensity
)
from .components.couplers import (
    IndependentCoupler,
    ChunkedSharedCoupler
)
from .components.networks import (
    ConstantNetwork,
    get_mlp,
    get_resnet,
    OneLayerCNN,
    VAEOneLayerDecoder,
    TupleMLP,
    TupleConvnetUpscaleVector,
    TupleResnetVectorizeImage,
    ConvEncoder,
    ConvDecoder
)


_DTYPE = torch.get_default_dtype()


def get_vi_density(
        schema,
        latent_shape,
        targets
):
    layer_config = schema[0]
    schema_tail = schema[1:]

    assert layer_config["type"] == "vi-head", f"Invalid layer {layer_config['type']}"

    vi_posterior = get_density_recursive(schema_tail, latent_shape)

    density = VIDensity(
        vi_posterior=vi_posterior,
        target=targets["model"],
        groundtruth_target=targets["groundtruth"]
    )
    return density


def get_density_recursive(
        schema,
        x_shape
):
    # TODO: We could specify this explicitly to allow different prior distributions
    if not schema:
        return get_standard_gaussian_density(x_shape=x_shape)

    layer_config = schema[0]
    schema_tail = schema[1:]

    if layer_config["type"] == "split":
        split_x_shape = (x_shape[0] // 2, *x_shape[1:])
        return SplitDensity(
            density_1=get_density_recursive(
                schema=schema_tail,
                x_shape=split_x_shape
            ),
            density_2=get_standard_gaussian_density(x_shape=split_x_shape),
            dim=1
        )

    elif layer_config["type"] == "vi-base":
        return get_vi_base_density(
            layer_config=layer_config,
            latent_shape=x_shape
        )

    else:
        return get_bijection_density(
            layer_config=layer_config,
            schema_tail=schema_tail,
            x_shape=x_shape
        )


def get_bijection_density(layer_config, schema_tail, x_shape):
    # XXX: For vi problems, x_shape is the shape of the latent dim
    bijection = get_bijection(layer_config=layer_config, x_shape=x_shape)

    prior = get_density_recursive(
        schema=schema_tail,
        x_shape=bijection.z_shape
    )

    if layer_config.get("num_u_channels", 0) == 0:
        return BijectionVIPosterior(
            bijection=bijection,
            vi_prior=prior
        )

    else:
        if layer_config["amortized"]:
            r_u_in_shape = (x_shape, layer_config["data_shape"])
        else:
            r_u_in_shape = x_shape

        return CifVIPosterior(
            bijection=bijection,
            vi_prior=prior,
            q_u_given_w=get_conditional_density(
                num_channels_per_output=layer_config["num_u_channels"],
                coupler_config=layer_config["p_coupler"],
                input_shape=x_shape
            ),
            r_u_given_z=get_conditional_density(
                num_channels_per_output=layer_config["num_u_channels"],
                coupler_config=layer_config["q_coupler"],
                input_shape=r_u_in_shape
            ),
            amortized=layer_config["amortized"]
        )


def get_standard_gaussian_density(x_shape):
    return DiagonalGaussianDensity(
        mean=torch.zeros(x_shape),
        log_stddev=torch.zeros(x_shape),
        num_fixed_samples=64
    )


def get_bijection(
        layer_config,
        x_shape
):
    if layer_config["type"] == "flatten":
        return ViewBijection(x_shape=x_shape, z_shape=(int(np.prod(x_shape)),))

    elif layer_config["type"] == "made":
        assert len(x_shape) == 1
        return MADEBijection(
            num_input_channels=x_shape[0],
            hidden_channels=layer_config["hidden_channels"],
            activation=get_activation(layer_config["activation"])
        )

    elif layer_config["type"] == "batch-norm":
        return BatchNormBijection(
            x_shape=x_shape,
            per_channel=layer_config["per_channel"],
            apply_affine=layer_config["apply_affine"],
            momentum=layer_config["momentum"]
        )

    elif layer_config["type"] == "cond-affine":
        return ConditionalAffineBijection(
            x_shape=x_shape,
            coupler=get_coupler(
                input_shape=(layer_config["num_u_channels"], *x_shape[1:]),
                num_channels_per_output=x_shape[0],
                config=layer_config["st_coupler"]
            )
        )

    elif layer_config["type"] == "flip":
        return FlipBijection(x_shape=x_shape, dim=1)

    elif layer_config["type"] == "linear":
        assert len(x_shape) == 1
        return LULinearBijection(num_input_channels=x_shape[0])

    elif layer_config["type"] == "rand-channel-perm":
        return RandomChannelwisePermutationBijection(x_shape=x_shape)

    elif layer_config["type"] == "nsf-ar":
        assert len(x_shape) == 1
        return AutoregressiveRationalQuadraticSplineBijection(
            num_input_channels=x_shape[0],
            num_hidden_layers=layer_config["num_hidden_layers"],
            num_hidden_channels=layer_config["num_hidden_channels"],
            num_bins=layer_config["num_bins"],
            tail_bound=layer_config["tail_bound"],
            activation=get_activation(layer_config["activation"]),
            dropout_probability=layer_config["dropout_probability"]
        )

    elif layer_config["type"] == "nsf-c":
        assert len(x_shape) == 1
        return CoupledRationalQuadraticSplineBijection(
            num_input_channels=x_shape[0],
            num_hidden_layers=layer_config["num_hidden_layers"],
            num_hidden_channels=layer_config["num_hidden_channels"],
            num_bins=layer_config["num_bins"],
            tail_bound=layer_config["tail_bound"],
            activation=get_activation(layer_config["activation"]),
            dropout_probability=layer_config["dropout_probability"],
            reverse_mask=layer_config["reverse_mask"]
        )

    else:
        assert False, f"Invalid layer type {layer_config['type']}"


def get_conditional_density(
        num_channels_per_output,
        coupler_config,
        input_shape
):
    return DiagonalGaussianConditionalDensity(
        coupler=get_coupler(
            input_shape=input_shape,
            num_channels_per_output=num_channels_per_output,
            config=coupler_config
        )
    )


def get_coupler(
        input_shape,
        num_channels_per_output,
        config
):
    if config["independent_nets"]:
        return get_coupler_with_independent_nets(
            input_shape=input_shape,
            num_channels_per_output=num_channels_per_output,
            shift_net_config=config["shift_net"],
            log_scale_net_config=config["log_scale_net"]
        )

    else:
        return get_coupler_with_shared_net(
            input_shape=input_shape,
            num_channels_per_output=num_channels_per_output,
            net_config=config["shift_log_scale_net"]
        )


def get_coupler_with_shared_net(
        input_shape,
        num_channels_per_output,
        net_config
):
    return ChunkedSharedCoupler(
        shift_log_scale_net=get_coupler_net(
            input_shape=input_shape,
            num_output_channels=2*num_channels_per_output,
            net_config=net_config
        )
    )


def get_coupler_with_independent_nets(
        input_shape,
        num_channels_per_output,
        shift_net_config,
        log_scale_net_config
):
    return IndependentCoupler(
        shift_net=get_coupler_net(
            input_shape=input_shape,
            num_output_channels=num_channels_per_output,
            net_config=shift_net_config
        ),
        log_scale_net=get_coupler_net(
            input_shape=input_shape,
            num_output_channels=num_channels_per_output,
            net_config=log_scale_net_config
        )
    )


def get_coupler_net(input_shape, num_output_channels, net_config):
    num_input_channels = input_shape[0]

    if net_config["type"] == "mlp":
        assert len(input_shape) == 1
        return get_mlp(
            num_input_channels=num_input_channels,
            hidden_channels=net_config["hidden_channels"],
            num_output_channels=num_output_channels,
            activation=get_activation(net_config["activation"])
        )

    elif net_config["type"] == "resnet":
        assert len(input_shape) == 3
        return get_resnet(
            num_input_channels=num_input_channels,
            hidden_channels=net_config["hidden_channels"],
            num_output_channels=num_output_channels
        )

    elif net_config["type"] == "constant":
        value = torch.full((num_output_channels, *input_shape[1:]), net_config["value"])
        return ConstantNetwork(value=value, fixed=net_config["fixed"])

    elif net_config["type"] == "identity":
        assert num_output_channels == num_input_channels
        return lambda x: x

    elif net_config["type"] == "amortized-coupler":
        assert len(input_shape) == 2
        if net_config["structure"] == "mlp":
            net = TupleMLP

        elif net_config["structure"] == "vector-to-image":
            net = TupleConvnetUpscaleVector

        elif net_config["structure"] == "image-to-vector":
            net = TupleResnetVectorizeImage

        else:
            assert False, f"Invalid net structure {net_config['structure']}"

        return net(
            input_shapes=input_shape,
            hidden_channels=net_config["hidden_channels"],
            num_output_channels=num_output_channels,
            activation=get_activation(net_config["activation"])
        )

    else:
        assert False, f"Invalid net type {net_config['type']}"


def get_activation(name):
    if name == "tanh":
        return nn.Tanh
    elif name == "relu":
        return nn.ReLU
    else:
        assert False, f"Invalid activation {name}"


def get_vi_base_density(layer_config, latent_shape):
    # XXX: Does not handle multi-indexed latents
    latent_dim = latent_shape[0]

    if layer_config["amortized"]:
        if layer_config.get("vae_one_layer", False):
            return BaseConditionalVIPosterior(
                prior_density=DiagonalGaussianConditionalDensity(
                    coupler=ChunkedSharedCoupler(
                        shift_log_scale_net=OneLayerCNN(
                            input_shape=layer_config["data_shape"],
                            output_dim=latent_shape[0]*2,
                            num_hidden_channels=layer_config["num_hidden_channels"],
                            kernel_size=layer_config["kernel_size"],
                            stride=layer_config["stride"],
                            activation=get_activation(layer_config["activation"])
                        )
                    )
                )
            )

        elif layer_config.get("vae_large", False):
            return BaseConditionalVIPosterior(
                prior_density=DiagonalGaussianConditionalDensity(
                    coupler=ChunkedSharedCoupler(
                        shift_log_scale_net=ConvEncoder(
                            context_features=latent_dim*2,
                            channels_multiplier=layer_config["channels_multiplier"]
                        )
                    )
                )
            )

        else:
            return BaseConditionalVIPosterior(
                prior_density=get_conditional_density(
                    num_channels_per_output=latent_dim,
                    coupler_config=layer_config["coupler"],
                    input_shape=layer_config["data_shape"]
                )
            )

    elif layer_config["target"] == "bayesian-neural-net":
        return BaseUnconditionalVIPosterior(
            prior_density=get_standard_gaussian_density(
                x_shape=latent_shape
            )
        )

    else:
        mean = torch.zeros(latent_shape, dtype=_DTYPE)
        log_stddev = np.log(layer_config["stddev"]) + torch.zeros(latent_shape, dtype=_DTYPE)
        return BaseUnconditionalVIPosterior(
            prior_density=DiagonalGaussianDensity(
                mean=mean.requires_grad_(layer_config["learnable_mean"]),
                log_stddev=log_stddev.requires_grad_(layer_config["learnable_stddev"])
            )
        )


def get_vi_targets(layer_config, latent_shape):
    if layer_config["target"] == "mog-lattice":
        groundtruth = get_model_gaussian_mixture_density(
            layer_config=layer_config,
            latent_shape=latent_shape,
            groundtruth=True
        )
        # XXX: No distinction between groundtruth and model here
        gt_and_model = UnconditionalVITarget(groundtruth)
        return {
            "groundtruth": gt_and_model,
            "model": gt_and_model
        }

    elif layer_config["target"] in ["vae-one-layer", "vae-large"]:
        prior = get_standard_gaussian_density(latent_shape)

        if layer_config["target"] == "vae-one-layer":
            likelihood = BernoulliConditionalDensity(
                coupler=VAEOneLayerDecoder(
                    latent_dim=latent_shape[0],
                    data_shape=layer_config["data_shape"],
                    num_hidden_channels=layer_config["num_hidden_channels"],
                    kernel_size=layer_config["kernel_size"],
                    stride=layer_config["stride"]
                )
            )

        else:
            likelihood = BernoulliConditionalDensity(
                coupler=ConvDecoder(
                    latent_features=latent_shape[0],
                    channels_multiplier=layer_config["channels_multiplier"]
                )
            )

        model = BayesianFactorModel(
            prior=prior,
            likelihood=likelihood
        )

        return {
            "groundtruth": None,
            "model": model
        }

    else:
        assert False, f"Invalid VI target {layer_config['target']}"


def get_nn_param_list(list_of_inits, requires_grad=False):
    list_of_params = [
        nn.Parameter(torch.tensor(p, dtype=_DTYPE), requires_grad=requires_grad)
        for p in list_of_inits
    ]
    return nn.ParameterList(list_of_params)
