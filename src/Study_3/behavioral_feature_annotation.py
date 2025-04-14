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
        "--datafile",
        default="../data/drive_suppress_data.csv",
    )

    parser.add_argument(
        "-d",
        "--diff_file",
        default="../results/linear_features/gemma-2-2b/diff_vectors.pkl",
        type=str,
        help="Where to find the file containin diff vectors",
    )

    args = parser.parse_args()
    return args


def featurize_data(model, diff_vectors, data):

    results_dict = {
        "sentence": [],
        "layer": [],
        "improbable_impossible_projection": [],
        "impossible_inconceivable_projection": [],
        "improbable_inconceivable_projection": [],
    }

    print("Featurizing Data")
    for n, stimulus in enumerate(data["sentence"]):
        print(n/len(data))
        tokens = model.to_tokens(stimulus, prepend_bos=True)
        _, cache = model.run_with_cache(tokens)

        for layer in range(model.cfg.n_layers):
            residual_stream = cache[utils.get_act_name("resid_post", layer)].cpu()[
                0, -1
            ]
            results_dict["sentence"].append(stimulus)
            results_dict["layer"].append(layer)

            for i, features in enumerate(CONCEPTS):
                feature_0, feature_1 = features.split("_")[0], features.split("_")[1]
                feature_vector = (
                    diff_vectors[feature_1][layer] - diff_vectors[feature_0][layer]
                )
                projection = torch.dot(feature_vector, residual_stream).item()
                results_dict[features + "_projection"].append(projection)

    data = pd.DataFrame.from_dict(results_dict)
    return data


if __name__ == "__main__":
    # Parse Args
    args = parse_arguments()

    CONCEPTS = ["improbable_impossible", "impossible_inconceivable", "improbable_inconceivable"]

    # Set up model
    torch.set_grad_enabled(False)
    model = transformer_lens.HookedTransformer.from_pretrained(
        args.model,
        device="cuda",
        dtype="bfloat16",
    )

    steering_vectors = pkl.load(open(args.diff_file, "rb"))
    data = pd.read_csv(args.datafile)
    # filter data
    data = data.groupby(["roi", "target_UID"])
    for id, group in data:
        data = group
        break
    data = data[
        [
            "sentence",
            "log-prob-5gram_mean",
            "log-prob-gpt2-xl_mean",
            "log-prob-pcfg_mean",
            "rating_arousal_mean",
            "rating_conversational_mean",
            "rating_sense_mean",
            "rating_gram_mean",
            "rating_frequency_mean",
            "rating_imageability_mean",
            "rating_others_thoughts_mean",
            "rating_physical_mean",
            "rating_places_mean",
            "rating_valence_mean",
        ]
    ]
    data.to_csv("../data/filtered_behavioral.csv")
    print(len(data))
    data = featurize_data(model, steering_vectors, data)
    data.to_csv("../data/annotated_behavioral.csv")
