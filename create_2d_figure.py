#!/usr/bin/env python3
import os
import sys
import json
import torch
import matplotlib.pyplot as plt

from cif.experiment import load_run
from cif.visualizer import TwoDimensionalVIVisualizer
from cif.writer import DummyWriter


_FIG_SIZE = (5, 5)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


try:
    run = sys.argv[1]
except IndexError:
    raise ValueError("Need to specify the run directory to the script")


metrics_file = "metrics.json"
try:
    with open(os.path.join(run, metrics_file), "r") as f:
        metrics = json.load(f)
    title = f"ELBO = {metrics['elbo']:.3f}"
except FileNotFoundError:
    raise ValueError(f"Could not locate `{metrics_file}` in directory `{run}`")


density, _, _, _, config, _ = load_run(run, device, False)

density.eval()
vis = TwoDimensionalVIVisualizer(
    writer=DummyWriter(run),
    fig_x_limits=config["fig_x_limits"],
    fig_y_limits=config["fig_y_limits"],
    title=title
)

plt.figure(figsize=_FIG_SIZE)
vis.visualize(density, epoch=0)
plt.savefig(os.path.join(run, "posterior_visualization.pdf"))
plt.close()
