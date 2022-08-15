from .dsl import group, base, provides, GridParams


group(
    "vi_mog",
    [
        "dummy-mog"
    ]
)


@base
def config(dataset, use_baseline):
    return {
        "amortized": False,

        "data_shape": (1,),
        "latent_dim": 2,
        "num_u_channels": 1,

        "use_cond_affine": not use_baseline,

        "batch_norm": False,

        "max_epochs": 20000,
        "max_grad_norm": None,
        "early_stopping": False,
        "max_bad_valid_epochs": 5000,
        "train_batch_size": 1000,
        "valid_batch_size": 1000,
        "test_batch_size": 10000,

        "opt": "adam",
        "lr": 1e-3,
        "lr_schedule": "none",
        "weight_decay": 0.,
        "epochs_per_test": 100,

        "num_valid_elbo_samples": 1,
        "num_test_elbo_samples": 100,

        "source_stddev": GridParams(0.1, 1, 10),

        "target": "mog-lattice",
        "target_components": 9,
        "target_stddev": 0.25,
        "target_limit": 2,
        "fig_x_limits": [-3, 3],
        "fig_y_limits": [-3, 3],
        # # NOTE: Uncomment below for K = 16 example
        # "target_components": 16,
        # "target_stddev": 0.25,
        # "target_limit": 3,
        # "fig_x_limits": [-4, 4],
        # "fig_y_limits": [-4, 4],

        "learnable_prior_mean": False,
        "learnable_prior_stddev": False,

        "st_nets": [10] * 2,
        "p_nets": [10] * 2,
        "q_nets": [10] * 2

    }


@provides("nsf")
def nsf(dataset, model, use_baseline):
    return {
        "schema_type": "nsf",
        "autoregressive": True,
        "use_linear": False,

        "max_grad_norm": 5,

        "num_density_layers": 5,
        "num_bins": 8,
        "num_hidden_channels": 32,
        "num_hidden_layers": 2,
        "tail_bound": 3,
        "dropout_probability": 0.
    }
