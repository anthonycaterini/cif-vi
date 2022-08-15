#!/usr/bin/env python3
import os
import argparse
import tqdm
import numpy as np
import torch

from cif.experiment import load_run
from cif.datasets import SupervisedDataset, get_loader


parser = argparse.ArgumentParser("ELBO Estimator Study")
parser.add_argument("-d", "--dir", type=str, required=True,
    help="Directory containing trained run.")
parser.add_argument("--m-fn", type=str, required=True, choices=["linear", "sqrt"],
    help="Specification of how M depends on N.")
args = parser.parse_args()


_NUM_SAMPLES_LIST = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000]
_RUNS_PER_SAMPLE = 20
_MAX_MEMORY = 4000000

run = args.dir
m_fn = args.m_fn
outfile_name = f"{m_fn}_elbo_estimator.txt" 
outfile_path = os.path.join(run, outfile_name)

marginal_avg_list = []
marginal_stddev_list = []
auxiliary_avg_list = []
auxiliary_stddev_list = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


try:
    density, _, _, _, _, _ = load_run(
        run_dir=run,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        data_parallel=False
    )
except KeyError as e:
    import ipdb; ipdb.set_trace()
    print("Error {0} for path {1}".format(e, run))

density.eval()


for n in _NUM_SAMPLES_LIST:
    print(f"n={n}")
    x_values = torch.rand((n, 1)).to(device)

    if m_fn == "linear":
        m = n
    elif m_fn == "sqrt":
        m = int(np.sqrt(n))
    else:
        assert False, f"Unrecognized m dependence {m_fn}"

    batch_size = min(n, _MAX_MEMORY // m)

    marginal_elbo_list = []
    auxiliary_elbo_list = []

    for k in range(_RUNS_PER_SAMPLE):
        print(f"\tRun {k+1} of {_RUNS_PER_SAMPLE}")
        elbo_result = density.elbo(x_values)
        z_samples = elbo_result["approx-posterior-sample"]
        auxiliary_elbo = elbo_result["elbo"].mean().item()

        x_z_dataset = SupervisedDataset("", "test", x_values, z_samples)
        x_z_loader = get_loader(x_z_dataset, device, batch_size, drop_last=False)

        sum_marginal_elbo = 0.
        for x, z in tqdm.tqdm(x_z_loader):
            sum_marginal_elbo += density.marginal_elbo(x, z, m)["marginal-elbo"].sum().item()
        marginal_elbo = sum_marginal_elbo / n

        marginal_elbo_list.append(marginal_elbo)
        auxiliary_elbo_list.append(auxiliary_elbo)

    marginal_avg_list.append(np.mean(marginal_elbo_list))
    marginal_stddev_list.append(np.std(marginal_elbo_list))
    auxiliary_avg_list.append(np.mean(auxiliary_elbo_list))
    auxiliary_stddev_list.append(np.std(auxiliary_elbo_list))


data_list_iterator = zip(
    _NUM_SAMPLES_LIST,
    marginal_avg_list,
    marginal_stddev_list,
    auxiliary_avg_list,
    auxiliary_stddev_list
)

with open(outfile_path, "w") as f:
    f.write(f"m dependence {m_fn}\n")
    f.write("n, \tmarginal_avg, \tauxiliary_avg\n")
    for n, m_avg, m_std, a_avg, a_std in data_list_iterator:
        f.write(f"{n}, \t{m_avg:.4f} +/- {m_std:.4f}, \t{a_avg:.4f} +/- {a_std:.4f}\n")
