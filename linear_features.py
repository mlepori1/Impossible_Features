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

SENTENCE_FEATURES = ["probable", "improbable", "impossible", "inconceivable", "syntactic"] #,"shuffled", "monsters", "birds"]

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
        "-k",
        "--k_folds",
        default=5,
        type=int,
        help="Number of CV folds"
    )
 
    args = parser.parse_args()
    return args


def args2dict(args):
    return {
        "model": args.model,
        "k_folds": args.k_folds,
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

def generate_mean_diff_vector(probable_states, feature_states, layer):
    probable_states = probable_states[layer]
    feature_states = feature_states[layer]
    diffs = [probable_states[i] - feature_states[i] for i in range(len(probable_states))]
    diffs = torch.stack(diffs, dim=0)
    mean_diff = torch.mean(diffs, dim=0)
    # Normalize to unit length
    normed_diff = mean_diff / torch.norm(mean_diff)
    return normed_diff

def train_eval(probable_states, feature_states, diff_vector, layer):
    # Featurize data using its projection onto the diff vector
    probable_states = torch.stack(probable_states[layer], dim=0)
    feature_states = torch.stack(feature_states[layer], dim=0)
    states = torch.cat([probable_states, feature_states], dim=0)
    projections = np.array([torch.dot(states[i], diff_vector).item() for i in range(len(states))])
    projections = projections.reshape(-1, 1)
    labels = [0] * len(probable_states) + [1] * len(feature_states)

    # Fit a simple classifier
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(projections, labels)

    # Return train accuracy
    train_acc = clf.score(projections, labels)

    return train_acc


def eval_diff_vector(model, train_feature2state, test_feature2state, layer2diffs):

    # Results: Layer -> Sentence Feature -> Test Acc
    results = defaultdict(dict)
    raw_features = defaultdict(dict)

    for feature_label in SENTENCE_FEATURES[1:]:
        for layer in range(model.cfg.n_layers):

            # Featurize data using its projection onto the diff vector
            probable_states = torch.stack(train_feature2state["probable"][layer], dim=0)
            feature_states = torch.stack(train_feature2state[feature_label][layer], dim=0)
            states = torch.cat([probable_states, feature_states], dim=0)

            diff_vector = layer2diffs[layer]
            projections = np.array([torch.dot(states[i], diff_vector).item() for i in range(len(states))])
            projections = projections.reshape(-1, 1)
            labels = [0] * len(probable_states) + [1] * len(feature_states)

            # Fit a simple classifier
            clf = KNeighborsClassifier(n_neighbors=3)
            clf.fit(projections, labels)

            # Get test split features
            probable_states = torch.stack(test_feature2state["probable"][layer], dim=0)
            feature_states = torch.stack(test_feature2state[feature_label][layer], dim=0)
            states = torch.cat([probable_states, feature_states], dim=0)

            projections = np.array([torch.dot(states[i], diff_vector).item() for i in range(len(states))])
            projections = projections.reshape(-1, 1)
            labels = [0] * len(probable_states) + [1] * len(feature_states)

            # Eval and store results
            test_acc = clf.score(projections, labels)
            results[layer][feature_label] = test_acc
            raw_features[layer][feature_label] = projections.reshape(-1)

    return results, raw_features

    
def linear_analysis(model, train_splits, test_split):
    # Test this on held out data
    # To understand how/whether features are represented as categorically different from one another/
    # with different linear representations, use this classifier to classify the test sets for other features, report this

    torch.set_grad_enabled(False)
    results_dict = {}
    # Store all diff vectors for representational geometry analysis
    diff_vectors = {}
    # Store all raw features for plotting projections
    all_raw_features = {}
    # Store all hidden states
    train_feature2states = {}
    test_feature2states = {}

    ## Compute and cache all hidden states for each sentence feature
    print(f"Caching States")
    train_feature2states["probable"] = compute_hidden_states(model, train_splits, "probable")
    train_feature2states["improbable"] = compute_hidden_states(model, train_splits, "improbable")
    train_feature2states["impossible"] = compute_hidden_states(model, train_splits, "impossible")
    train_feature2states["inconceivable"] = compute_hidden_states(model, train_splits, "inconceivable")
    train_feature2states["syntactic"] = compute_hidden_states(model, train_splits, "syntactic")

    test_feature2states["probable"] = compute_hidden_states(model, [test_split], "probable")
    test_feature2states["improbable"] = compute_hidden_states(model, [test_split], "improbable")
    test_feature2states["impossible"] = compute_hidden_states(model, [test_split], "impossible")
    test_feature2states["inconceivable"] = compute_hidden_states(model, [test_split], "inconceivable")
    test_feature2states["syntactic"] = compute_hidden_states(model, [test_split], "syntactic")

    for sentence_feature in SENTENCE_FEATURES[1:]: # Compute mean diff vector btw probable and each other feature
        print(f"Processing: {sentence_feature}")
        results_dict[sentence_feature] = {}
        diff_vectors[sentence_feature] = {}

        # Store per-layer diff vectors and train accs
        layer2diff = {}
        layer2trainacc = {}

        for layer in range(model.cfg.n_layers):
            print(f"Layer: {layer}")
            mean_diff_vector = generate_mean_diff_vector(train_feature2states["probable"], train_feature2states[sentence_feature], layer)
            train_acc = train_eval(train_feature2states["probable"], train_feature2states[sentence_feature], mean_diff_vector, layer)
            diff_vectors[sentence_feature][layer] = mean_diff_vector
            layer2diff[layer] = mean_diff_vector
            layer2trainacc[layer] = train_acc
            print(f"Train Accuracy: {train_acc}")

        # Featurize and Eval iterates over layers and tests all features, 
        # returning a dictionary of results per layer
        print("Evaluating Diff Vector")
        test_results, raw_features = eval_diff_vector(model, train_feature2states, test_feature2states, layer2diff)
        all_raw_features[sentence_feature] = raw_features

        # Collect results
        for layer in range(model.cfg.n_layers):
            results_dict[sentence_feature][layer] = test_results[layer]
            results_dict[sentence_feature][layer]["train"] = layer2trainacc[layer]
    
    return results_dict, diff_vectors, all_raw_features

def reformat_results(results):
    # Results: A list of dicts: sentence_feature -> layer -> trains acc, all splits test acc

    results_df = {
        "train_feature": [],
        "fold": [],
        "layer": [],
        "condition": [],
        "accuracy": [],
    }

    for split_idx, split_dict in enumerate(results):
        for train_feature in split_dict.keys():
            for layer in split_dict[train_feature].keys():
                for condition in split_dict[train_feature][layer].keys():
                    results_df["train_feature"].append(train_feature)
                    results_df["fold"].append(split_idx)
                    results_df["layer"].append(layer)
                    results_df["condition"].append(condition)
                    results_df["accuracy"].append(
                        split_dict[train_feature][layer][condition]
                    )
    return pd.DataFrame.from_dict(results_df)

def plot_linear_analysis(results, figfolder, model_name):
    # Plot train acc over layers
    train_acc_data = results[results["condition"] == "train"]
    g = sns.catplot(
        data=train_acc_data, x="layer", y="accuracy", col="train_feature",
        kind="bar", height=4, aspect=.6,
    )
    g.set_axis_labels("Layers", "Accuracy")
    g.set_titles("{col_name}")
    g.set(ylim=(0, 1))
    g.despine(left=True)
    # add overhead title
    g.fig.subplots_adjust(top=0.8) 
    g.fig.suptitle(f"{model_name}: Train Accuracies by Layer")
    plt.xticks(fontsize=8, rotation=90)
    plt.savefig(os.path.join(figfolder, "train_accs.png"), bbox_inches="tight")
    plt.figure()

    # For each layer, plot train/test/generalization acc for all different diff vectors
    eval_folder = os.path.join(figfolder, "Evals")
    os.makedirs(eval_folder, exist_ok=True)
    for layer in results["layer"].unique():
        layer_data = results[results["layer"] == layer]
        g = sns.catplot(
            data=layer_data, x="condition", y="accuracy", col="train_feature",
            kind="bar", height=4, aspect=.6,
        )
        g.set_xticklabels(rotation=45)
        g.set_axis_labels("", "Accuracy")
        g.set_titles("{col_name}")
        g.set(ylim=(0, 1))
        g.despine(left=True)
        # add overhead title
        g.fig.subplots_adjust(top=0.8) 
        g.fig.suptitle(f"{model_name} Layer {layer} Evaluations")
        plt.savefig(os.path.join(eval_folder, f"layer_{layer}_eval.png"), bbox_inches="tight")
        plt.figure()

def reformat_diff_vectors(diff_vectors):
    # diff_vectors: A list of dicts: sentence_feature -> layer -> diff_vector, all splits test acc
    # Want a list of lists of diff vectors, ordered by feature, and a list of labels

    reordered_diffs = []
    n_layers = len(diff_vectors[0]["improbable"].keys())
    sentence_features = list(diff_vectors[0].keys())
    for layer in range(n_layers):
        diff_list = []
        label_list = []
        for sentence_feature in sentence_features:
            for split_idx, split_dict in enumerate(diff_vectors):
                diff_list.append(split_dict[sentence_feature][layer].to(torch.float16).cpu())
                label_list.append(sentence_feature + f"_{str(split_idx)}")
        reordered_diffs.append(diff_list)
    return np.stack(reordered_diffs, axis=0), label_list

def plot_diff_vectors(diffs, label_list, figfolder, model_name):

    # For each layer, plot the RSM of all computed mean diff vectors
    rsm_folder = os.path.join(figfolder, "RSM")
    os.makedirs(rsm_folder, exist_ok=True)
    for i in range(len(diffs)):

        similarities = cosine_similarity(diffs[i])
        plt.imshow(similarities)
        plt.colorbar()
        plt.xticks(range(len(similarities)), labels=label_list, rotation=90, fontsize=10)
        plt.yticks(range(len(similarities)), labels=label_list, fontsize=10)
  
        plt.title(f"{model_name} Layer {i} Mean-Diff RSM")
        plt.savefig(os.path.join(rsm_folder, f"layer_{i}_rsm.png"), bbox_inches="tight")
        plt.figure()

def reformat_features(features):

    # Features: layer -> feature label -> projections
    features_df = {
        "Vector Type": [],
        "Eval Condition": [],
        "Projection": [],
        "layer": [],
    }

    for vector_type, layer2eval in features.items():
        for layer, type2projections in layer2eval.items():
            for stim_type, projections in type2projections.items():
                features_df["Vector Type"] += [vector_type] * len(projections)
                features_df["Eval Condition"] += [stim_type] * len(projections)
                features_df["layer"] += [layer] * len(projections)
                features_df["Projection"] += projections.tolist()
    return pd.DataFrame.from_dict(features_df).reset_index(drop=True)

def plot_features(features, figfolder, model_name):
    # For each layer, plot the raw projections
    projections_folder = os.path.join(figfolder, "Projections")
    os.makedirs(projections_folder, exist_ok=True)
    for layer in features["layer"].unique():
        layer_data = features[features["layer"] == layer]
        g = sns.stripplot(
            data=layer_data, x="Vector Type", y="Projection", hue="Eval Condition", jitter=.3
        )
        plt.xticks(rotation=45)
        plt.title(f"{model_name} Layer {layer} Projections on Vector")
        plt.savefig(os.path.join(projections_folder, f"layer_{layer}_projections.png"), bbox_inches="tight", dpi=500)
        plt.figure()


if __name__ == "__main__":
    # Parse Args
    args = parse_arguments()

    ### Set Up Output ###
    outfolder = os.path.join(args.outfolder, args.model)
    figfolder = os.path.join(outfolder, "figures")
    os.makedirs(outfolder, exist_ok=True)
    os.makedirs(figfolder, exist_ok=True)

    # Write config
    with open(os.path.join(outfolder, "cfg.json"), "w") as f:
        json.dump(args2dict(args), f)

    # Set up model
    torch.set_grad_enabled(False)
    model = transformer_lens.HookedTransformer.from_pretrained(
        args.model,
        device="cuda",
        dtype="bfloat16",
    )

    # Run Analysis
    dataset = SentenceFeaturesDataset(file="stimuli")
    fold_proportion = 1.0/args.k_folds
    splits = random_split(dataset, [fold_proportion] * args.k_folds)

    k_fold_results = []
    k_fold_diff_vectors = []
    k_fold_features = []
    for i in range(args.k_folds):
        print(f"Fold {i}\n++++++++++++")
        test_split = splits[i]
        train_idxs = list(range(args.k_folds))
        train_idxs.remove(i)
        train_splits = [splits[idx] for idx in train_idxs]

        results, diff_vectors, raw_features = linear_analysis(model, train_splits, test_split)
        k_fold_results.append(results)
        k_fold_diff_vectors.append(diff_vectors)
        k_fold_features.append(raw_features)

    results = reformat_results(k_fold_results)
    results.to_csv(os.path.join(outfolder, "results.csv"))

    pkl.dump(k_fold_diff_vectors, open(os.path.join(outfolder, "diff_vectors.pkl"), "wb"))

    # Plot results
    plot_linear_analysis(results, figfolder, args.model)

    # Plot RSM of Diff Vectors
    diff_vectors, label_list = reformat_diff_vectors(k_fold_diff_vectors)
    plot_diff_vectors(diff_vectors, label_list, figfolder, args.model)

    # Plot sample scatterplot of raw features
    raw_features = reformat_features(k_fold_features[0])
    plot_features(raw_features, figfolder, args.model)