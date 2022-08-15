def get_schema(config):
    schema = get_base_schema(config=config)

    if config["use_cond_affine"]:
        assert config["num_u_channels"] > 0
        schema = add_cond_affine_before_each_normalise(schema=schema, config=config)

    schema = apply_pq_coupler_config_settings(schema=schema, config=config)

    if config["batch_norm"]:
        schema = replace_normalise_with_batch_norm(schema=schema, config=config)
    else:
        schema = remove_layers_of_given_type(schema=schema, layer_type="normalise")

    schema = edit_schema_for_vi(schema=schema, config=config)

    return schema


# TODO: Could just pass the whole config to each constructor
def get_base_schema(config):
    ty = config["schema_type"]

    if ty == "maf":
        return get_maf_schema(
            num_density_layers=config["num_density_layers"],
            hidden_channels=config["ar_map_hidden_channels"]
        )

    elif ty == "nsf":
        return get_nsf_schema(config=config)

    elif ty == "vae":
        return []

    else:
        assert False, f"Invalid schema type `{ty}'"


def remove_non_normalise_layers(schema):
    return [layer for layer in schema if layer["type"] == "normalise"]


def remove_layers_of_given_type(schema, layer_type):
    return [layer for layer in schema if layer["type"] != layer_type]


def replace_normalise_with_batch_norm(schema, config):
    if config["batch_norm_use_running_averages"]:
        new_schema = []
        momentum = config["batch_norm_momentum"]

    else:
        new_schema = [
            {
                "type": "passthrough-before-eval",
                # XXX: This should be sufficient for most of the non-image
                # datasets we have but can be made a config value if necessary
                "num_passthrough_data_points": 100_000
            }
        ]
        momentum = 1.

    for layer in schema:
        if layer["type"] == "normalise":
            new_schema.append({
                "type": "batch-norm",
                "per_channel": True, # Hard coded for now; seems always to do better
                "momentum": momentum,
                "apply_affine": config["batch_norm_apply_affine"]
            })

        else:
            new_schema.append(layer)

    return new_schema


def add_cond_affine_before_each_normalise(schema, config):
    new_schema = []
    flattened = False
    for layer in schema:
        if layer["type"] == "flatten":
            flattened = True
        elif layer["type"] == "normalise":
            new_schema.append(get_cond_affine_layer(config, flattened))

        new_schema.append(layer)

    return new_schema


def apply_pq_coupler_config_settings(schema, config):
    new_schema = []
    flattened = False
    for layer in schema:
        if layer["type"] == "flatten":
            flattened = True

        if layer.get("num_u_channels", 0) > 0:
            layer = {
                **layer,
                "p_coupler": get_p_coupler_config(config, flattened),
                "q_coupler": get_q_coupler_config(config, flattened)
            }

        new_schema.append(layer)

    return new_schema


def get_cond_affine_layer(config, flattened):
    return {
        "type": "cond-affine",
        "num_u_channels": config["num_u_channels"],
        "st_coupler": get_st_coupler_config(config, flattened),
    }


def get_st_coupler_config(config, flattened):
    return get_coupler_config("t", "s", "st", config, flattened)


def get_p_coupler_config(config, flattened):
    return get_coupler_config("p_mu", "p_sigma", "p", config, flattened)


def get_q_coupler_config(config, flattened):
    return get_coupler_config("q_mu", "q_sigma", "q", config, flattened)


def get_coupler_config(
        shift_prefix,
        log_scale_prefix,
        shift_log_scale_prefix,
        config,
        flattened
):
    shift_key = f"{shift_prefix}_nets"
    log_scale_key = f"{log_scale_prefix}_nets"
    shift_log_scale_key = f"{shift_log_scale_prefix}_nets"

    if shift_key in config and log_scale_key in config:
        assert shift_log_scale_key not in config, "Over-specified coupler config"
        return {
            "independent_nets": True,
            "shift_net": get_coupler_net_config(config[shift_key], flattened),
            "log_scale_net": get_coupler_net_config(config[log_scale_key], flattened)
        }

    elif shift_log_scale_key in config:
        assert shift_key not in config and log_scale_key not in config, \
                "Over-specified coupler config"

        if config.get("amortized", False) and shift_log_scale_prefix == "q":
            return {
                "independent_nets": False,
                "shift_log_scale_net": get_amortized_net_confg(config[shift_log_scale_key])
            }

        else:
            return {
                "independent_nets": False,
                "shift_log_scale_net": get_coupler_net_config(config[shift_log_scale_key], flattened)
            }

    else:
        assert False, f"Must specify either `{shift_log_scale_key}', or both `{shift_key}' and `{log_scale_key}'"


def get_amortized_net_confg(net_spec):
    return {
        "type": "amortized-coupler",
        "activation": net_spec["activation"],
        "hidden_channels": net_spec["hidden_channels"],
        "structure": net_spec["structure"]
    }


def get_coupler_net_config(net_spec, flattened):
    if net_spec in ["fixed-constant", "learned-constant"]:
        return {
            "type": "constant",
            "value": 0,
            "fixed": net_spec == "fixed-constant"
        }

    elif net_spec == "identity":
        return {
            "type": "identity"
        }

    elif isinstance(net_spec, list):
        if flattened:
            return {
                "type": "mlp",
                "activation": "tanh",
                "hidden_channels": net_spec
            }
        else:
            return {
                "type": "resnet",
                "hidden_channels": net_spec
            }

    elif isinstance(net_spec, int):
        if flattened:
            return {
                "type": "mlp",
                "activation": "tanh",
                # Multiply by 2 to match the 2 hidden layers of the glow-cnns
                "hidden_channels": [net_spec] * 2
            }
        else:
            return {
                "type": "glow-cnn",
                "num_hidden_channels": net_spec,
                "zero_init_output": True
            }

    else:
        assert False, f"Invalid net specifier {net_spec}"


def get_maf_schema(
        num_density_layers,
        hidden_channels
):
    result = [{"type": "flatten"}]

    for i in range(num_density_layers):
        if i > 0:
            result.append({"type": "flip"})

        result += [
            {
                "type": "made",
                "hidden_channels": hidden_channels,
                "activation": "tanh"
            },
            {
                "type": "normalise"
            }
        ]

    return result


def get_nsf_schema(
        config
):
    result = [{"type": "flatten"}]

    for i in range(config["num_density_layers"]):
        if "use_linear" in config and not config["use_linear"]:
            result += [{"type": "rand-channel-perm"}]
        else:
            result += [{"type": "rand-channel-perm"}, {"type": "linear"}]

        layer = {
            "type": "nsf-ar" if config["autoregressive"] else "nsf-c",
            "num_hidden_channels": config["num_hidden_channels"],
            "num_hidden_layers": config["num_hidden_layers"],
            "num_bins": config["num_bins"],
            "tail_bound": config["tail_bound"],
            "activation": "relu",
            "dropout_probability": config["dropout_probability"]
        }

        if not config["autoregressive"]:
            layer["reverse_mask"] = i % 2 == 0

        result.append(layer)

        result.append(
            {
                "type": "normalise"
            }
        )

    if "use_linear" in config and not config["use_linear"]:
        result += [{"type": "rand-channel-perm"}]
    else:
        result += [{"type": "rand-channel-perm"}, {"type": "linear"}]

    return result


def edit_schema_for_vi(schema, config):
    schema = remove_layers_of_given_type(schema, "flatten")
    schema = add_key_to_all_layers(schema, "amortized", config["amortized"])
    schema = add_key_to_all_layers(schema, "data_shape", config["data_shape"])

    vi_head_layer = get_vi_head_layer(config)
    vi_base_layer = get_vi_base_layer(config)

    return [vi_head_layer, *schema, vi_base_layer]


def get_vi_head_layer(config):
    layer_config = {}
    layer_config["type"] = "vi-head"
    layer_config["target"] = config["target"]
    layer_config["data_shape"] = config["data_shape"]

    if layer_config["target"] == "mog-lattice":
        layer_config["components"] = config["target_components"]
        layer_config["stddev"] = config["target_stddev"]
        layer_config["lim"] = config["target_limit"]

        layer_config["learnable_mixture_weights"] = False
        layer_config["learnable_mixture_means"] = False
        layer_config["learnable_mixutre_stddevs"] = False

    elif layer_config["target"] == "vae-one-layer":
        layer_config["num_hidden_channels"] = config["num_vae_hidden_channels"]
        layer_config["kernel_size"] = config["kernel_size"]
        layer_config["stride"] = config["stride"]

    elif layer_config["target"] == "vae-large":
        layer_config["channels_multiplier"] = config["vae_channels_multiplier"]

    else:
        assert False, f"Invalid target {layer_config['target']}"

    return layer_config


def get_vi_base_layer(config):
    layer_config = {}
    layer_config["type"] = "vi-base"
    layer_config["amortized"] = config["amortized"]
    layer_config["data_shape"] = config["data_shape"]
    layer_config["target"] = config["target"]

    if layer_config["amortized"]:
        if "encoder" in config:
            if config["encoder"] == "vae-one-layer":
                layer_config["num_hidden_channels"] = config["num_vae_hidden_channels"]
                layer_config["kernel_size"] = config["kernel_size"]
                layer_config["stride"] = config["stride"]
                layer_config["vae_one_layer"] = True
                layer_config["activation"] = "tanh"

            elif config["encoder"] == "vae-large":
                layer_config["channels_multiplier"] = config["vae_channels_multiplier"]
                layer_config["vae_large"] = True

        else:
            layer_config["coupler"] = get_coupler_config(
                "base_mu", "base_sigma", "base", config, True
            )

    else:
        layer_config["stddev"] = config["source_stddev"]
        layer_config["learnable_mean"] = config["learnable_prior_mean"]
        layer_config["learnable_stddev"] = config["learnable_prior_stddev"]

    return layer_config


def add_key_to_all_layers(schema, key, value):
    for layer in schema:
        layer[key] = value
    return schema
