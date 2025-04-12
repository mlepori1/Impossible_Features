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
        "-d",
        "--diff_file",
        default="../results/linear_features/gemma-2-2b/diff_vectors.pkl",
        type=str,
        help="Where to find the file containin diff vectors"
    )
 
    args = parser.parse_args()
    return args


    
def vocabulary_projection(model, diff_vectors):
    
    for i, features in enumerate(CONCEPTS):
        print(features, "\n###################")
        for layer in range(model.cfg.n_layers):
            print("Layer ", layer)
            feature_0, feature_1 = features.split("_")[0], features.split("_")[1]
            feature_vector = diff_vectors[feature_1][layer] - diff_vectors[feature_0][layer]
            print("Driven vocab items")
            output_logits = model.unembed(feature_vector.to("cuda"))
            print(model.to_string(torch.topk(output_logits, 30).indices))
            print("Suppressed vocab items")
            output_logits = model.unembed(-1 * feature_vector.to("cuda"))
            print(model.to_string(torch.topk(output_logits, 30).indices)) 
            



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
    vocabulary_projection(model, steering_vectors)
