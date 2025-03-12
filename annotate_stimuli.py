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
import itertools
import scipy.stats as stats

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
        "--outfile",
        default="./data/annotated_stimuli",
        type=str,
        help="File to put results",
    )

    parser.add_argument(
        "--compute_features",
        default=False,
        action="store_true",
        help="Whether to compute features",
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


def create_diff_vectors(improb, imposs, inc):

    return {
        "improbable_impossible": imposs - improb,
        "impossible_inconceivable": inc - imposs,
    }


def zscore(results_dict, feature):
    mean  = np.mean(results_dict[feature])
    std = np.std(results_dict[feature])
    results_dict[feature] = list((np.array(results_dict[feature]) - mean)/std)

def normalize(results_dict):

    normed_feats = SENTENCE_FEATURES
    for feat in normed_feats:
        results_dict[feat] = np.clip(results_dict[feat], a_min=None, a_max=0.0)
        results_dict[feat] = results_dict[feat] / np.min(results_dict[feat])

    return results_dict

def preprocess_line(line):
    # Get rid of numeral indices preceding lines
    line = line.split(" ")
    if line[0][-1] == ".":
        line = " ".join(line[1:])
    else:
        line = " ".join(line)
    return line

if __name__ == "__main__":
    # Parse Args
    args = parse_arguments()

    # Set up model
    if args.compute_features is True:
        torch.set_grad_enabled(False)
        model = transformer_lens.HookedTransformer.from_pretrained(
            args.model,
            device="cuda",
            dtype="bfloat16",
        )

        # Layer at which to compute metrics
        layer=12
        steering_vectors = pkl.load(open(f"./results/linear_features/{args.model}/diff_vectors.pkl", "rb"))[0]
        steering_vectors = create_diff_vectors(steering_vectors["improbable"][layer], steering_vectors["impossible"][layer], steering_vectors["inconceivable"][layer])

        SENTENCE_FEATURES = ["improbable_impossible", "impossible_inconceivable"]
        GENERATION_CLASS = ["improbable", "impossible", "inconceivable"]

        torch.set_grad_enabled(False)

        results_dict = {
            "prompt": [],
            "base_class": [],
            "surprisal": [],
            "improbable_impossible": [],
            "impossible_inconceivable": [],
        }

        print("Featurizing Data")
        for cls in GENERATION_CLASS:
            print(f"Featurizing {cls}")
            generated_stimuli = open(f"./data/generated_{cls}.txt", "r")

            for line in generated_stimuli.readlines():
                # Just process real generations, not fluff
                if ":" not in line and line.strip() != "":
                    # Compute average surprisal over full sentence
                    line = line.strip()
                    line = preprocess_line(line)
                    tokens = model.to_tokens(line, prepend_bos=True)
                    labels = tokens[0, 1:]
                    logits, cache =  model.run_with_cache(tokens)
                    logits = logits[0, :-1]

                    logprobs = F.log_softmax(logits, dim=-1)
                    logprobs = logprobs[range(len(labels)), labels]
                    surprisal = torch.mean(-1 * logprobs).item()

                    results_dict["prompt"].append(line)
                    results_dict["base_class"].append(cls)
                    results_dict["surprisal"].append(surprisal)
                    residual_stream = cache[utils.get_act_name("resid_post", layer)].cpu()[0, -1]

                    for i, feature in enumerate(SENTENCE_FEATURES):
                        # Unit norm the residual stream
                        residual_stream = residual_stream / torch.norm(residual_stream)
                        feature_vector = steering_vectors[feature]
                        projection = torch.dot(feature_vector, residual_stream).item()
                        results_dict[feature].append(projection)


        data = pd.DataFrame.from_dict(results_dict)
        
        # Find controversial and non-controversial stimuli
        class_dfs = []

        # IMPROBABLE
        class_data = data[data["base_class"] == "improbable"]
        low_thresh = class_data["improbable_impossible"].quantile(0.2)
        high_thresh = class_data["improbable_impossible"].quantile(0.8)

        noncontroversial_data = class_data[class_data["improbable_impossible"] > high_thresh]
        controversial_data = class_data[class_data["improbable_impossible"] < low_thresh]

        noncontroversial_data["class"] = ["improbable"] * len(noncontroversial_data)
        controversial_data["class"] = ["improbable-impossible"] * len(controversial_data)
        class_dfs.append(noncontroversial_data)
        class_dfs.append(controversial_data)

        # IMPOSSIBLE
        class_data = data[data["base_class"] == "impossible"]
        low_thresh = class_data["impossible_inconceivable"].quantile(0.2)
        high_thresh = class_data["improbable_impossible"].quantile(0.8)

        nc_low_thresh = class_data["impossible_inconceivable"].quantile(0.4)
        nc_high_thresh = class_data["improbable_impossible"].quantile(0.6)

        controversial_data_low = class_data[class_data["impossible_inconceivable"] < low_thresh] # More inconceivable
        controversial_data_high = class_data[class_data["improbable_impossible"] > high_thresh] # More improbable
        noncontroversial_data = class_data[(class_data["impossible_inconceivable"] > nc_low_thresh) & (class_data["improbable_impossible"] < nc_high_thresh)]

        noncontroversial_data["class"] = ["impossible"] * len(noncontroversial_data)
        controversial_data_low["class"] = ["impossible-inconceivable"] * len(controversial_data_low)
        controversial_data_high["class"] = ["improbable-impossible"] * len(controversial_data_high)

        class_dfs.append(controversial_data_high)
        class_dfs.append(noncontroversial_data)
        class_dfs.append(controversial_data_low)

        # INCONCEIVABLE
        class_data = data[data["base_class"] == "inconceivable"]
        low_thresh = class_data["impossible_inconceivable"].quantile(0.2)
        high_thresh = class_data["impossible_inconceivable"].quantile(0.8)

        controversial_data = class_data[class_data["impossible_inconceivable"] > high_thresh]
        noncontroversial_data = class_data[class_data["impossible_inconceivable"] < low_thresh]

        noncontroversial_data["class"] = ["inconceivable"] * len(noncontroversial_data)
        controversial_data["class"] = ["impossible-inconceivable"] * len(controversial_data)
        class_dfs.append(controversial_data)
        class_dfs.append(noncontroversial_data)

        data = pd.concat(class_dfs)
        data.to_csv(args.outfile + "_unfiltered.csv")

    if not args.compute_features:
        data = pd.read_csv(args.outfile + "_unfiltered.csv")

    # Sample to ensure surprisal equivalence between all {improb, imposs, improb-imposs} and
    # {imposs, inc, imposs-inc} 
    ENTRIES_PER_CLASS = 10
    MAX_ITER = 50000
    ALPHA_THRESHOLD = 0.1

    class_groupings = [
        ["improbable", "impossible", "improbable-impossible"],
        ["impossible", "inconceivable", "impossible-inconceivable"]
        ]
    
    for class_labels in class_groupings:
        print(f"Computing T-Tests: {class_labels[-1]}")

        for _ in range(MAX_ITER):
            dfs = [data[data["class"] == label].sample(ENTRIES_PER_CLASS) for label in class_labels]
            curr_df = pd.concat(dfs)

            # Compute all pairwise T-tests
            all_pass = True
            p_vals = []
            for label1, label2 in itertools.combinations(class_labels, 2):
                t_stat, p_value = stats.ttest_ind(
                    curr_df[curr_df["class"] == label1]["surprisal"], 
                    curr_df[curr_df["class"] == label2]["surprisal"], 
                )
                if p_value < ALPHA_THRESHOLD:  # If distributions are significantly different, reject this sample
                    all_pass = False
                    break
                else:
                    p_vals.append(p_value)
            
            # If all pairwise T-tests are NOT significant, accept this sample
            if all_pass:
                print(f"Identified a good distribution: P-vals: {p_vals}")
                pass_df = curr_df
                break  # Stop if we found a good sample set

        if not all_pass:
            print("No subsampling succeeded")
        else:
            pass_df.to_csv(args.outfile + "_" + class_labels[-1] + ".csv")
