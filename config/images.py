from .dsl import group, base, provides, GridParams


group(
    "images",
    [
        "mnist",
        "fashion-mnist",
        "cifar10",
        "svhn"
    ]
)


@base
def config(dataset, use_baseline):
    return {
        "num_u_channels": 1,
        "use_cond_affine": True,

        "batch_norm": False,
        "batch_norm_apply_affine": use_baseline,
        "batch_norm_use_running_averages": True,
        "batch_norm_momentum": 0.1,

        "lr_schedule": "none",
        "max_bad_valid_epochs": 50,
        "max_grad_norm": None,
        "max_epochs": 1000,
        "epochs_per_test": 1,
        "early_stopping": True,

        "num_valid_elbo_samples": 5,
        "num_test_elbo_samples": 10
    }


def vae_base(dataset):
    return {
        "amortized": True,

        "schema_type": "vae",

        # XXX: Hacky
        "data_shape": (3,32,32) if dataset in ["svhn", "cifar10"] else (1,28,28),

        "num_u_channels": 2,

        "train_batch_size": 100,
        "valid_batch_size": 500,
        "test_batch_size": 500,

        "opt": "adam",
        "lr": 1e-3,
        "lr_schedule": "none",
        "weight_decay": 0.,
        "epochs_per_test": 5,

        "num_valid_elbo_samples": 1,
        "num_test_elbo_samples": 100,
    }


@provides("small-vae-vi")
def small_vae_vi(dataset, model, use_baseline):
    vae_base_cfg = vae_base(dataset)
    return {
        **vae_base_cfg,

        "latent_dim": 20,

        "target": "vae-one-layer",
        "encoder": "vae-one-layer",
        "num_vae_hidden_channels": 8,
        "kernel_size": 4,
        "stride": 2
    }


@provides("large-vae-vi")
def large_vae_vi(dataset, model, use_baseline):
    vae_base_cfg = vae_base(dataset)
    return {
        **vae_base_cfg,

        "latent_dim": 32,

        "target": "vae-large",
        "encoder": "vae-large",
        "vae_channels_multiplier": 16,

        "valid_batch_size": 100,
        "test_batch_size": 50
    }


@provides("large-dec-small-enc")
def large_dec_small_enc(dataset, model, use_baseline):
    vae_base_cfg = vae_base(dataset)
    return {
        **vae_base_cfg,

        "latent_dim": 32,

        "target": "vae-large",
        "vae_channels_multiplier": 16,

        "encoder": "vae-one-layer",
        "num_vae_hidden_channels": 8,
        "kernel_size": 4,
        "stride": 2,

        "valid_batch_size": 100,
        "test_batch_size": 50
    }


@provides("large-enc-small-dec")
def large_enc_small_dec(dataset, model, use_baseline):
    vae_base_cfg = vae_base(dataset)
    return {
        **vae_base_cfg,

        "latent_dim": 20,

        "target": "vae-one-layer",
        "num_vae_hidden_channels": 8,
        "kernel_size": 4,
        "stride": 2,

        "encoder": "vae-large",
        "vae_channels_multiplier": 16,

        "valid_batch_size": 100,
        "test_batch_size": 100
    }


def nsf_base(dataset):
    return {
        "schema_type": "nsf",
        "autoregressive": True,
        "use_linear": True,

        "num_bins": 8,
        "num_hidden_layers": 2,
        "tail_bound": 3,
        "dropout_probability": 0.,

        "st_nets": [10] * 2,
        "p_nets": [10] * 2,
    }


@provides("nsf-ldse")
def nsf_ldse(dataset, model, use_baseline):
    vi_base_cfg = large_dec_small_enc(dataset, model, use_baseline)
    nsf_base_cfg = nsf_base(dataset)
    return {
        **vi_base_cfg,
        **nsf_base_cfg,

        "max_grad_norm": 5,

        "num_density_layers": 10,
        "num_hidden_channels": GridParams(32,44) if use_baseline else 32,

        "q_nets": {
            "activation": "tanh",
            "hidden_channels": 8,
            "structure": "vector-to-image"
        }
    }


@provides("nsf-vi")
def nsf_vi(dataset, model, use_baseline):
    vi_base_cfg = small_vae_vi(dataset, model, use_baseline)
    nsf_base_cfg = nsf_base(dataset)
    return {
        **vi_base_cfg,
        **nsf_base_cfg,

        "max_grad_norm": 5,

        "num_density_layers": 10,
        "num_hidden_channels": 44 if use_baseline else 32,

        "q_nets": {
            "activation": "tanh",
            "hidden_channels": 8,
            "structure": "vector-to-image"
        }
    }


@provides("nsf-vi-durkan")
def nsf_vi_durkan(dataset, model, use_baseline):
    vi_base_cfg = large_vae_vi(dataset, model, use_baseline)
    nsf_base_cfg = nsf_base(dataset)
    return {
        **vi_base_cfg,
        **nsf_base_cfg,

        "max_grad_norm": None,

        "num_density_layers": 5,
        "num_hidden_channels": 128 if use_baseline else 96,

        "q_nets": {
            "activation": "relu",
            "hidden_channels": [10, 50],
            "structure": "image-to-vector"
        }
    }


def maf_base(use_baseline):
    return {
        "schema_type": "maf",

        "batch_norm": use_baseline,

        "num_density_layers": 5,
        "ar_map_hidden_channels": [512] * 2 if use_baseline else [420] * 2,

        "st_nets": [128] * 2,
        "p_nets": [128] * 2,
        "q_nets": {
            "activation": "tanh",
            "hidden_channels": 64,
            "structure": "vector-to-image"
        }
    }


@provides("maf-vi")
def maf_vi(dataset, model, use_baseline):
    vi_base_cfg = small_vae_vi(dataset, model, use_baseline)
    maf_base_cfg = maf_base(use_baseline)
    return {
        **vi_base_cfg,
        **maf_base_cfg
    }


@provides("maf-ldse")
def maf_ldse(dataset, model, use_baseline):
    vi_base_cfg = large_dec_small_enc(dataset, model, use_baseline)
    maf_base_cfg = maf_base(use_baseline)
    return {
        **vi_base_cfg,
        **maf_base_cfg
    }
