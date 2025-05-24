"""
For each stimulus analyzed in this experiment, record the projection of that
stimulus on the difference vectors derived from the LLM
"""

import argparse
import os
import json
from collections import defaultdict
import pickle as pkl
from functools import partial
import itertools
import scipy.stats as stats

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import torch
import transformer_lens
import transformer_lens.utils as utils

import torch.nn.functional as F


def plot_partition(data, threshold):

    partition = data[data["rating"] < threshold]
    plt.figure()
    sns.scatterplot(
        data=partition,
        x="improbable_impossible",
        y="impossible_inconceivable",
        hue="rating",
        palette=sns.color_palette("viridis", as_cmap=True),
        alpha=0.5,
        linewidth=0,
    )
    plt.title("Sparsely Compositional Plot")
    plt.savefig("./sparse_partition.png")
    return partition


if __name__ == "__main__":
    # Parse Args
    data = pd.read_csv("sparse_data.csv")
    partition = plot_partition(data, -2.5)

    quantiles = partition.quantile([.5, .1, .9], numeric_only=True)
    
    # Get low inconceivable stimuli
    low_inc = partition[partition["impossible_inconceivable"] < quantiles["impossible_inconceivable"].iloc[1]]
    high_inc = partition[partition["impossible_inconceivable"] > quantiles["impossible_inconceivable"].iloc[2]]


    with open("./sparse_stims.txt", "w") as f:
        f.write("Low Inconceivability, High Impossibility\n")
        for stim in low_inc["stimulus"]:
            f.write(stim + "\n")
        f.write("High Inconceivability, High Impossibility\n")
        for stim in high_inc["stimulus"]:
            f.write(stim + "\n")
    print(low_inc["stimulus"])
    print(high_inc["stimulus"])
