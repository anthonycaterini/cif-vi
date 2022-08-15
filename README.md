# README

Code release for [Variational inference with continuously-indexed normalizing flows](https://proceedings.mlr.press/v161/caterini21a.html), which appeared at UAI 2021.

This codebase is essentially a fork of an older version of the [codebase](https://github.com/jrmcornish/cif) for [Relaxing bijectivity constraints with continuously indexed normalising flows](https://proceedings.mlr.press/v119/cornish20a.html), which appeared at ICML 2020.

## Usage

### Setup

First, install submodules:

    $ git submodule init
    $ git submodule update

Next, install dependencies. If you use `conda`, the following will create an environment called `cif-vi`:

    conda env create -f env-lock.yml

Activate this with

    conda activate cif-vi

before running any code or tests.

If you don't use `conda`, then please see `environment.yml` for a list of required packages, which will need to be installed manually via `pip` etc.

## Mixture of Gaussians Example

To reproduce the results in **Section 4.1** of the paper, run the following command:

    ./main.py --model nsf --dataset dummy-mog [--baseline]

where the `--baseline` tag is optional, and indicates that we are running a baseline (non-CIF) configuration.
This will launch three runs of the $K=9$ experiment - one each for $\sigma_0 \in \{0.1, 1, 10\}$.
More runs with different initializations can be launched by adding the `--num-seeds` argument to the above command.

By default, this will create a directory `runs/`, which will contain Tensorboard logs giving various information about the training run, including 2D KDE plots in this case.
To inspect this, ensure you have `tensorboard` installed (e.g. `pip install tensorboard`), and run in a new terminal:

    tensorboard --logdir runs/ --port=8008

Keep this running, and navigate to http://localhost:8008, where the results should be visible.

The plots from each of the runs will be visible in Tensorboard.
Alternatively, if you want to create a PDF plot with a title for a run stored in `<run>`, first generate test metrics for this run using the command

    ./main.py --resume <run> --test

which will create a file `metrics.json` within the folder `<run>`(**which itself will show the ELBO for the run**). This will also brint test metrics to the screen. Then, create the plot `posterior_visualization.pdf` in the folder `<run>` using the command

    ./create_2d_figure.py <run>

The same workflow can be done for the $K=16$ example, but first **uncomment lines 50-55** in `config/vi_mog.py`.

We can also launch experiments with trainable $\sigma_0$, by adding the flags `--config learnable_prior_stddev=True --config source_stddev=1` at the command line.

Indeed, adding the `--config` flag allows updating the config values at the command line; another way to update the experiment specification is to go directly into the `config/vi_mog.py` file and manually updating.

### Estimator statistics

To test the ELBO estimator as in **Table 6** in the Appendix, let us first assume that a saved CIF run in located in the directory `<run>`.
Then, launch the command

    ./assess_elbo_estimator.py -d <run> --m-fn [linear|sqrt]

where `m-fn` describes the relationship between `M` and `N` (either linear or square-root).
This will produce a table `<m-fn>_elbo_estimator.txt` within `<run>` like the one in the Appendix, for a single choice of `m-fn`.

## Larger-Scale Experiments

**Instructions coming soon**

## Utilities

To inspect the model (either CIF or baseline) used for a given dataset, add the `--print-model` argument. To try out alternative configurations, simply modify the relevant options in `config.py`.


## Bibtex

    @inproceedings{caterini2021variational,
    title={Variational inference with continuously-indexed normalizing flows},
    author={Caterini, Anthony and Cornish, Rob and Sejdinovic, Dino and Doucet, Arnaud},
    booktitle={Uncertainty in Artificial Intelligence},
    pages={44--53},
    year={2021},
    organization={PMLR}
    }
