"""
This file takes the difference between SAE activations for particular sentence features,
runs a statistical test to determine whether any features are consistently different between
probable and feature'd sentences, then prints these features
"""

import argparse
import os
import json
from collections import defaultdict
import requests

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity

import torch
from sae_lens import HookedSAETransformer, SAE

import scipy.stats as stats

from data import SentenceFeaturesDataset

SENTENCE_FEATURES = ["probable", "improbable", "impossible", "inconceivable", "syntactic", "shuffled", "monsters"]

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
        default="./results/linear_features",
        type=str,
        help="Folder to put results",
    )

    parser.add_argument(
        "-l",
        "--sae_layers",
        default=[12],
        help="layers to put the SAE into"
    )
 
    args = parser.parse_args()
    return args


def args2dict(args):
    return {
        "model": args.model,
        "sae_layers": args.sae_layers,
    }

def setup_autointerp(layer):
    # Retrieve the AutoInterp descriptions of each feature in a given SAE layer from Neuronpedia

    url = f"https://www.neuronpedia.org/api/explanation/export?modelId=gemma-2-2b&saeId={layer}-gemmascope-res-16k"
    headers = {"Content-Type": "application/json"}

    response = requests.get(url, headers=headers)
    data = response.json()
    explanations_df = pd.DataFrame(data)
    # rename index to "feature"
    explanations_df.rename(columns={"index": "feature"}, inplace=True)
    explanations_df["description"] = explanations_df["description"].apply(
        lambda x: x.lower()
    )
    return explanations_df

def compute_sae_states(model, dataset, feature_label, sae_layers):
    layer2states = defaultdict(list)

    for datum in dataset:
        prompt = datum[feature_label]
        tokens = model.to_tokens(prompt, prepend_bos=True)
        _, cache = model.run_with_cache(tokens)

        for layer in sae_layers:
            # Extract the final SAE state in seq
            layer2states[layer].append(cache[
            f"blocks.{layer}.hook_resid_post.hook_sae_acts_post"
        ].to(torch.float16).cpu()[0, -1])
  
    return layer2states

def generate_vector_distributions(probable_states, feature_states, layer):
    probable_states = probable_states[layer]
    feature_states = feature_states[layer]

    probables = [probable_states[i] for i in range(len(probable_states))]
    probables = torch.stack(probables, dim=0)
    features = [feature_states[i] for i in range(len(feature_states))]
    features = torch.stack(features, dim=0)

    return probables, features

def get_testable_features(prob_dist, feature_dist):
    # Get nonzero feature count for bonferonni correction
    count = 0
    feature_idxs = []

    all_vectors = torch.concat([prob_dist, feature_dist], dim=0)

    for feat in range(all_vectors.shape[1]):
        if torch.any(all_vectors[:, feat] > 0):
            count+=1
            feature_idxs.append(feat)
    return count, feature_idxs

def run_statistical_test(prob_dist, feature_dist, threshold=.1):
    significant = []
    p_vals = []

    number_of_tests, feature_idxs = get_testable_features(prob_dist, feature_dist)
    print(number_of_tests)
    corrected_threshold = threshold/number_of_tests
    print(corrected_threshold)

    for feat in range(prob_dist.shape[1]):
        _, p_val = stats.wilcoxon(prob_dist[:, feat], y=feature_dist[:, feat], alternative="less")
        if p_val < corrected_threshold:
            significant.append(1)
            p_vals.append(p_val)
        else:
            significant.append(0)

    return np.nonzero(significant)[0], p_vals, len(significant)

def autointerp_features(significant_features, autointerp):
    descriptions = []
    for feature in significant_features:
        print(str(feature))
        descriptions.append(autointerp[autointerp["feature"] == str(feature)].iloc[0][
            "description"
        ]
        )

    return descriptions
        
def record_results(sentence_feature, layer, significant_features, total_features, autointerp_descriptions, p_vals, outpath):
    file = open(outpath, "w")
    file.write(f"{sentence_feature} - Layer {layer}\n")
    file.write(f"{len(significant_features)}/{total_features} are significant\n")
    for i, desc in enumerate(autointerp_descriptions):
        file.write(f"P-val: {p_vals[i]} Description: {desc}\n")
    file.close()

def sae_analysis(model, dataset, sae_layers, autointerp_dict, outfolder):
    # Generate latent SAE vectors for probable/featured sentences
    # Take difference between these vectors.
    # Run a statistical test on each dimension to see if it is significantly different from 0
    # Print out the features associated with those significant dimensions

    results_folder = os.path.join(outfolder, "sae_features")
    os.makedirs(results_folder, exist_ok=True)

    probable_sae_states = compute_sae_states(model, dataset, "probable", sae_layers)
    for sentence_feature in SENTENCE_FEATURES[1:]: # Compute mean diff vector btw probable and each other feature
        print(f"Processing: {sentence_feature}")

        feature_sae_states = compute_sae_states(model, dataset, sentence_feature, sae_layers)

        for layer in sae_layers:
            print(f"Layer: {layer}")
            prob_distribution, feature_distribution = generate_vector_distributions(probable_sae_states, feature_sae_states, layer)
            significant_features, p_vals, total_features = run_statistical_test(prob_distribution, feature_distribution)
            print(significant_features)
            autointerp_descriptions = autointerp_features(significant_features, autointerp_dict[layer])
            outpath = os.path.join(results_folder, f"{sentence_feature}_L{layer}.txt")
            record_results(sentence_feature, layer, significant_features, total_features, autointerp_descriptions, p_vals, outpath)

if __name__ == "__main__":
    # Parse Args
    args = parse_arguments()

    ### Set Up Output ###
    outfolder = os.path.join(args.outfolder, args.model)
    os.makedirs(outfolder, exist_ok=True)

    # Write config
    with open(os.path.join(outfolder, "cfg.json"), "w") as f:
        json.dump(args2dict(args), f)

    # Set up model
    # Set up model
    torch.set_grad_enabled(False)
    model = HookedSAETransformer.from_pretrained_no_processing(
        args.model,
        device="cuda",
        dtype=torch.bfloat16,
    )

    # Add all SAEs to the transformer
    saes = {}
    autointerp_dict = {}
    for layer in args.sae_layers:
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release="gemma-scope-2b-pt-res-canonical",
            sae_id=f"layer_{layer}/width_16k/canonical",
            device="cpu",
        )
        sae.to(device="cuda", dtype=torch.bfloat16)  # Convert SAE to bfloat16
        sae.use_error_term = True

        saes[layer] = sae
        model.add_sae(sae)

        autointerp_dict[layer] = setup_autointerp(layer)

    # Run Analysis
    dataset = SentenceFeaturesDataset(file="stimuli")
    sae_analysis(model, dataset, args.sae_layers, autointerp_dict, outfolder)
       