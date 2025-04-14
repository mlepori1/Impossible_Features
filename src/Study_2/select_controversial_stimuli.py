"""
Algorithm:
1) Compute vectors representing the difference between impossible and improbable, impossible and inconceivable
2) For each generated stimulus, project it onto these two axes, and plot the projections
3) Compute the centroid value of each of the categories
4) Define a controversial stimulus as one that is most in-between two centroids, 
    noncontroversial as a stimulus farthest away from the two other centroids
5) Generate datasets containing 20 stimuli that fall in each of 
    {improbable, impossisble, inconceivable, controversial-improb-imposs, controversial-imposs-inc}
6) Sample stimuli to control for the distribution of surprisal, creating a set for {improb, controversial-improb-imposs, impossible}
and another for {impossible, controversial-imposs-inc, inconceivable}
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
        "--outdir",
        default="./data/experiment_3/",
        type=str,
        help="File to put results",
    )

    parser.add_argument(
        "--layer",
        default=12,
        type=int,
        help="Layer to use for vectors",
    )

    parser.add_argument(
        "--compute_features",
        default=False,
        action="store_true",
        help="Whether to compute features",
    )

    parser.add_argument(
        "--datafile",
        default="./data/experiment_3/annotated_stimuli.csv",
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


def min_max_scale(results_dict):
    for feat in SENTENCE_FEATURES:
        features = results_dict[feat] 
        results_dict[feat] = (features - np.min(features))/(np.max(features) - np.min(features))
    return results_dict

def preprocess_line(line):
    # Get rid of numeral indices preceding lines
    line = line.split(" ")
    line = " ".join(line[1:])
    return line

def featurize_data(args):
    
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
        generated_stimuli = open(f"data/experiment_3/generated_{cls}.txt", "r")

        for line in generated_stimuli.readlines():
            if line.strip() != "" and line.split()[0][-1] == "." and ":" not in line:
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
                    feature_vector = steering_vectors[feature]
                    projection = torch.dot(feature_vector, residual_stream).item()
                    results_dict[feature].append(projection)

    data = pd.DataFrame.from_dict(results_dict)
    data = min_max_scale(data)
    data.to_csv(os.path.join(args.outdir, "annotated_stimuli.csv"), index=False)
    return data

def plot_projections(data, args):

    improbable = data[data["base_class"] == "improbable"]
    impossible = data[data["base_class"] == "impossible"]
    inconceivable = data[data["base_class"] == "inconceivable"]

    plt.scatter(x=improbable["improbable_impossible"], y=improbable["impossible_inconceivable"], c="red", alpha=0.5, label="improbable")
    plt.scatter(x=np.median(improbable["improbable_impossible"]), y=np.median(improbable["impossible_inconceivable"]), c="red", marker='*', s=125, edgecolors="black")

    plt.scatter(x=impossible["improbable_impossible"], y=impossible["impossible_inconceivable"], c="blue", alpha=0.5, label="impossible")
    plt.scatter(x=np.median(impossible["improbable_impossible"]), y=np.median(impossible["impossible_inconceivable"]), c="blue", marker='*', s=125, edgecolors="black")

    plt.scatter(x=inconceivable["improbable_impossible"], y=inconceivable["impossible_inconceivable"], c="yellow", alpha=0.5, label="inconceivable")
    plt.scatter(x=np.median(inconceivable["improbable_impossible"]), y=np.median(inconceivable["impossible_inconceivable"]), c="gold", marker='*', s=125, edgecolors="black")

    plt.title("Generated Stimuli Projections")
    plt.legend()
    plt.xlabel("Improbable-Impossible")
    plt.ylabel("Impossible-Inconceivable")
    plt.savefig(os.path.join(args.outdir, "projection_plot.png"))

def generate_controversial(data, centroid_1, centroid_2, feature, n_controversial):
    """Stimuli are controversial if they minimize the distance
    from both centroids. 
    Controversial score = Max(Euclidean Distance(stim, centroid_1),
    Euclidean Distance(stim, centroid_2))
    """
    centroid_1_dist = np.abs(data[feature] - centroid_1)
    centroid_2_dist = np.abs(data[feature] - centroid_2)
    scores = np.max(np.stack([centroid_1_dist, centroid_2_dist]), axis=0)
    threshold = np.sort(scores)[n_controversial]
    return scores < threshold

def generate_noncontroversial(data, key_centroid, not_centroid, feature, n_noncontroversial):
    """Stimuli are noncontroversial if they maximize distance from the other centroid
    and minimize distance from centroid of interest.
    Noncontroversial score = Euclidean Distance(stim, not_centroid)
    - Euclidean Distance(stim, centroid)
    """
    centroid_dist = np.abs(data[feature] - key_centroid)
    not_centroid_dist = np.abs(data[feature] - not_centroid)
    scores = not_centroid_dist - centroid_dist
    threshold = np.sort(scores)[-n_noncontroversial]
    return scores >= threshold

def generate_dataset(data, centroid_1, centroid_2, feature):
    labels = feature.split("_")
    if feature == "improbable_impossible":
        centroid_idx = 0
    else:
        centroid_idx = 1
    
    data = data[data["base_class"].isin(labels)]
    data[f"{feature}_controversial"] = generate_controversial(data, centroid_1[centroid_idx], centroid_2[centroid_idx], feature, n_controversial=20)
    data[f"{feature}_{labels[0]}"] = generate_noncontroversial(data, centroid_1[centroid_idx], centroid_2[centroid_idx], feature, n_noncontroversial=20)
    data[f"{feature}_{labels[1]}"] = generate_noncontroversial(data, centroid_2[centroid_idx], centroid_1[centroid_idx], feature, n_noncontroversial=20)

    return data

def sample_data(data, feature, args):

    ENTRIES_PER_CLASS = 10
    MAX_ITER = 5000
    ALPHA_THRESHOLD = 0.1

    class_groupings = {
        "improbable_impossible": ["improbable_impossible_improbable", "improbable_impossible_impossible", "improbable_impossible_controversial"],
        "impossible_inconceivable": ["impossible_inconceivable_impossible", "impossible_inconceivable_inconceivable", "impossible_inconceivable_controversial"]
    }
    
    class_labels = class_groupings[feature]
    print(f"Computing T-Tests: {class_labels[-1]}")

    for _ in range(MAX_ITER):
        dfs = []
        for label in class_labels:
            label_df = data[data[label] == True].sample(ENTRIES_PER_CLASS)
            label_df["class"] = [label] * len(label_df)
            dfs.append(label_df)

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
        pass_df.to_csv(os.path.join(args.outdir, f"{feature}.csv"), index=False)

if __name__ == "__main__":
    # Parse Args
    args = parse_arguments()

    SENTENCE_FEATURES = ["improbable_impossible", "impossible_inconceivable"]
    GENERATION_CLASS = ["improbable", "impossible", "inconceivable"]

    # Set up model
    if args.compute_features is True:
        torch.set_grad_enabled(False)
        model = transformer_lens.HookedTransformer.from_pretrained(
            args.model,
            device="cuda",
            dtype="bfloat16",
        )

        # Layer at which to compute metrics
        layer=args.layer
        steering_vectors = pkl.load(open(f"./results/linear_features/{args.model}/diff_vectors.pkl", "rb"))[0]
        steering_vectors = create_diff_vectors(steering_vectors["improbable"][layer], steering_vectors["impossible"][layer], steering_vectors["inconceivable"][layer])

        # Featurize Data
        data = featurize_data(args)
    else:
        print("Loading Data")
        data = pd.read_csv(args.datafile)

    # Plot projections
    plot_projections(data, args)

    # Compute centroid coordinates using median
    improbable = data[data["base_class"] == "improbable"]
    impossible = data[data["base_class"] == "impossible"]
    inconceivable = data[data["base_class"] == "inconceivable"]

    improbable_centroid = (np.median(improbable["improbable_impossible"]), np.median(improbable["impossible_inconceivable"]))
    impossible_centroid = (np.median(impossible["improbable_impossible"]), np.median(impossible["impossible_inconceivable"]))
    inconceivable_centroid = (np.median(inconceivable["improbable_impossible"]), np.median(inconceivable["impossible_inconceivable"]))

    # Find controversial and non-controversial stimuli
    improb_imposs = generate_dataset(data, improbable_centroid, impossible_centroid, "improbable_impossible")
    imposs_inc = generate_dataset(data, improbable_centroid, impossible_centroid, "impossible_inconceivable")

    #Sample stimulus sets while controlling for surprisal
    sample_data(improb_imposs, "improbable_impossible", args)
    sample_data(imposs_inc, "impossible_inconceivable", args)