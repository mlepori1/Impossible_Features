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
        default="./results/naturalistic_eval",
        type=str,
        help="Folder to put results",
    )

    parser.add_argument(
        "-d",
        "--diff_file",
        default=5,
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


def compute_hidden_states(model, splits, feature_label):
    layer2states = defaultdict(list)

    for split in splits:
        for datum in split:
            prompt = datum[feature_label]
            tokens = model.to_tokens(prompt, prepend_bos=True)
            _, cache = model.run_with_cache(tokens)

            for layer in range(model.cfg.n_layers):
                # Extract the final hidden state
                layer2states[layer].append(cache[utils.get_act_name("resid_post", layer)].cpu()[0, -1])

    return layer2states
