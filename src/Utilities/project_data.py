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


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        default="gemma-2-2b",
        help="Model to use as backbone for analysis",
    )

    parser.add_argument(
        "--layer",
        default=12,
    )

    parser.add_argument(
        "--shades_of_zero",
        default="../data/stimuli_with_syntax.csv",
    )

    parser.add_argument(
        "--sparsely_compositional",
        default="../data/composlang-beh_no-dupes_agg-by-item.csv",
    )

    parser.add_argument(
        "-d",
        "--diff_file",
        default="../results/linear_features/gemma-2-2b/diff_vectors.pkl",
        type=str,
        help="Where to find the file containing diff vectors",
    )

    args = parser.parse_args()
    return args


def preprocess_shades(data):

    df = {"stimulus": [], "class": []}

    for cls in ["probable", "improbable", "impossible", "inconceivable"]:
        for _, row in data.iterrows():
            prefix = row["classification_prefix"]
            suffix = row[cls]
            stimulus = prefix + " " + suffix + "."
            stimulus = stimulus.replace("[POSS]", "their")
            df["stimulus"].append(stimulus)
            df["class"].append(cls)

    return pd.DataFrame.from_dict(df)


def preprocess_sparse(data):

    df = {"stimulus": [], "rating": []}

    for i, row in data.iterrows():
        stimulus = row["item"]
        if stimulus.lower()[0] in ["a", "e", "i", "o", "u"]:
            prefix = "There is an"
        else:
            prefix = "There is a"
        stimulus = prefix + " " + stimulus + "."
        rating = row["rating"]
        df["stimulus"].append(stimulus)
        df["rating"].append(rating)

    return pd.DataFrame.from_dict(df)


def featurize_data(model, layer, diff_vectors, data):

    layers = []
    improbable_impossible_projection = []
    impossible_inconceivable_projection = []

    print("Featurizing Data")
    for i, row in data.iterrows():
        print(i / len(data))
        stimulus = row["stimulus"]
        tokens = model.to_tokens(stimulus, prepend_bos=True)
        _, cache = model.run_with_cache(tokens)

        residual_stream = cache[utils.get_act_name("resid_post", layer)].cpu()[0, -1]
        layers.append(layer)

        for i, features in enumerate(CONCEPTS):
            feature_0, feature_1 = features.split("_")[0], features.split("_")[1]
            feature_vector = (
                diff_vectors[feature_1][layer] - diff_vectors[feature_0][layer]
            )
            projection = torch.dot(feature_vector, residual_stream).item()

            if features == "improbable_impossible":
                improbable_impossible_projection.append(projection)
            elif features == "impossible_inconceivable":
                impossible_inconceivable_projection.append(projection)

    data["layers"] = layers
    data["improbable_impossible"] = improbable_impossible_projection
    data["impossible_inconceivable"] = impossible_inconceivable_projection
    return data


def plot_shades(data):
    plt.figure()
    sns.scatterplot(
        data=data, x="improbable_impossible", y="impossible_inconceivable", hue="class"
    )
    plt.title("Shades of Zero Data Plot")
    plt.savefig("../results/retrospective/shades_projections.png")


def plot_sparse(data):
    plt.figure()
    sns.scatterplot(
        data=data,
        x="improbable_impossible",
        y="impossible_inconceivable",
        hue="rating",
        palette=sns.color_palette("viridis", as_cmap=True),
        alpha=0.5,
        linewidth=0,
    )
    plt.title("Sparsely Compositional Plot")
    plt.savefig("../results/retrospective/sparse_projections.png")


if __name__ == "__main__":
    # Parse Args
    args = parse_arguments()

    CONCEPTS = ["improbable_impossible", "impossible_inconceivable"]

    # Set up model
    torch.set_grad_enabled(False)
    model = transformer_lens.HookedTransformer.from_pretrained(
        args.model,
        device="cuda",
        dtype="bfloat16",
    )

    steering_vectors = pkl.load(open(args.diff_file, "rb"))

    # Prepare shades of zero data
    shades_data = pd.read_csv(args.shades_of_zero)
    shades_data = preprocess_shades(shades_data)
    # Featurize
    shades_data = featurize_data(model, args.layer, steering_vectors, shades_data)
    shades_data.to_csv("../results/retrospective/shades_data.csv")
    plot_shades(shades_data)

    # Prepare sparse compositional data
    sparse_data = pd.read_csv(args.sparsely_compositional)
    sparse_data = preprocess_sparse(sparse_data)
    # Featurize
    sparse_data = featurize_data(model, args.layer, steering_vectors, sparse_data)
    sparse_data.to_csv("../results/retrospective/sparse_data.csv")
    plot_sparse(sparse_data)
