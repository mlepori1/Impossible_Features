"""
This file runs an LM over prompts exhibiting different
sentence-level features, and computes the surprisal of the
sentences. Then, it creates a graph similar to Fig 3 of 
Shades of Zero
"""

import argparse
import os
import json
from collections import defaultdict
import pickle as pkl
from functools import partial

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity

import torch
import transformer_lens
import transformer_lens.utils as utils

import torch.nn.functional as F
from torch.utils.data import random_split

from data import SentenceFeaturesDataset

DIFF_FEATURES = ["improbable", "impossible", "inconceivable", "syntactic"]

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        default="gemma-2-2b",
        help="Model to use as backbone for analysis",
    )

    parser.add_argument(
        "--outfolder",
        default="./results/steering",
        type=str,
        help="Folder to put results",
    )

    parser.add_argument(
        "-d",
        "--diff_file",
        default="./results/linear_features/gemma-2-2b/diff_vectors.pkl",
        type=str,
        help="Where to find the file containin diff vectors for every split, layer, and feature"
    )
 
    args = parser.parse_args()
    return args


def args2dict(args):
    return {
        "model": args.model,
        "diff_file": args.diff_file,
    }


def steering_vector_hook(
    resid,
    hook,
    vector,
    multiplier,
) -> torch.Tensor:

    resid += vector.to("cuda") * multiplier
    return resid

if __name__ == "__main__":
    # Parse Args
    args = parse_arguments()

    # Set up model
    torch.set_grad_enabled(False)
    model = transformer_lens.HookedTransformer.from_pretrained(
        args.model,
        device="cuda",
        dtype="bfloat16",
    )

    layer=12
    steering_vectors = pkl.load(open(f"./results/linear_features/{args.model}/diff_vectors.pkl", "rb"))[0]

    SENTENCE_FEATURES = ["improbable", "impossible", "inconceivable", "syntactic"]

    for feature in SENTENCE_FEATURES:
        for multiplier in [-150, -100, -50, 0, 50, 100, 150]:
            print(f"{feature}: Multiplier {multiplier}")
            fwd_hooks = []

            fwd_hooks.append(
                (
                    f"blocks.{layer}.hook_resid_post", #Make the change on the value vector
                    partial(steering_vector_hook, vector=steering_vectors[feature][layer], multiplier=multiplier)
                )
            )
            logits = model.run_with_hooks(
                "peeling a potato with a",
                return_type="logits",
                fwd_hooks=fwd_hooks
            )

            top_k_indices = torch.topk(logits[0, -1], 20).indices
            print(model.to_string(top_k_indices))



