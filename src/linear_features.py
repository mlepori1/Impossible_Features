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

SENTENCE_FEATURES = ["probable", "improbable", "impossible", "inconceivable"]


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
        "-k", "--k_folds", default=5, type=int, help="Number of CV folds"
    )

    args = parser.parse_args()
    return args


def args2dict(args):
    return {
        "model": args.model,
        "k_folds": args.k_folds,
    }


def compute_hidden_states(model, splits, feature_label):
    """Extract the hidden states of the final token (".")
    from every sentence exhibiting a particular feature from
    every split.
    """
    layer2states = defaultdict(list)

    for split in splits:
        for datum in split:
            prompt = datum[feature_label]
            tokens = model.to_tokens(prompt, prepend_bos=True)
            _, cache = model.run_with_cache(tokens)

            for layer in range(model.cfg.n_layers):
                # Extract the final hidden state
                layer2states[layer].append(
                    cache[utils.get_act_name("resid_post", layer)].cpu()[0, -1]
                )

    return layer2states


def generate_mean_diff_vector(probable_states, feature_states, layer):
    """Compute the diff vector between a FEATURE hidden state and
    the corresponding probable hidden state. Take the mean of these
    diff vectors
    """
    probable_states = probable_states[layer]
    feature_states = feature_states[layer]
    diffs = [
        feature_states[i] - probable_states[i] for i in range(len(probable_states))
    ]
    diffs = torch.stack(diffs, dim=0)
    mean_diff = torch.mean(diffs, dim=0)
    return mean_diff


def eval_diff_vector(model, train_feature2state, test_feature2state, layer2diffs):
    """Compute two evaluations of a mean-diff vector, per layer.
    First, understand how much the projection along the mean diff vector
    partitions each feature from "probable" using a 2-way KNN classifier.

    Next, understand how much the projection along the mean diff vector
    partitions each of the classes from one another, using a 4-way classifier.
    """

    # Results: Layer -> Two-way -> Sentence Feature -> Binary Test Acc
    #                -> Four-way -> True Sentence Feature -> Predicted Sentence Feature -> Proportion
    results = defaultdict(dict)

    for layer in range(model.cfg.n_layers):
        # First, compute binary classifier results
        results[layer] = {}
        results[layer]["Two-Way"] = {}
        results[layer]["Four-Way"] = {}

        for feature_label in SENTENCE_FEATURES[1:]:

            # Featurize data using its projection onto the diff vector
            probable_states = torch.stack(train_feature2state["probable"][layer], dim=0)
            feature_states = torch.stack(
                train_feature2state[feature_label][layer], dim=0
            )
            states = torch.cat([probable_states, feature_states], dim=0)

            diff_vector = layer2diffs[layer]
            projections = np.array(
                [torch.dot(states[i], diff_vector).item() for i in range(len(states))]
            )
            projections = projections.reshape(-1, 1)
            labels = [0] * len(probable_states) + [1] * len(feature_states)

            # Fit a simple classifier
            clf = KNeighborsClassifier(n_neighbors=3)
            clf.fit(projections, labels)

            # Get test split features
            probable_states = torch.stack(test_feature2state["probable"][layer], dim=0)
            feature_states = torch.stack(
                test_feature2state[feature_label][layer], dim=0
            )
            states = torch.cat([probable_states, feature_states], dim=0)

            projections = np.array(
                [torch.dot(states[i], diff_vector).item() for i in range(len(states))]
            )
            projections = projections.reshape(-1, 1)
            labels = [0] * len(probable_states) + [1] * len(feature_states)

            # Eval and store results
            test_acc = clf.score(projections, labels)
            results[layer]["Two-Way"][feature_label] = test_acc

        # Second, compute four-way classifier results
        states = []
        for feature_label in SENTENCE_FEATURES:
            # Featurize data using its projection onto the diff vector
            states.append(torch.stack(train_feature2state[feature_label][layer], dim=0))
        states = torch.cat(states, dim=0)

        diff_vector = layer2diffs[layer]
        projections = np.array(
            [torch.dot(states[i], diff_vector).item() for i in range(len(states))]
        )
        projections = projections.reshape(-1, 1)
        labels = (
            [0] * int(len(states) / 4)
            + [1] * int(len(states) / 4)
            + [2] * int(len(states) / 4)
            + [3] * int(len(states) / 4)
        )

        # Fit a simple classifier
        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(projections, labels)

        states = []
        for feature_label in SENTENCE_FEATURES:
            # Featurize data using its projection onto the diff vector
            states.append(torch.stack(train_feature2state[feature_label][layer], dim=0))
        states = torch.cat(states, dim=0)

        diff_vector = layer2diffs[layer]
        projections = np.array(
            [torch.dot(states[i], diff_vector).item() for i in range(len(states))]
        )
        projections = projections.reshape(-1, 1)
        labels = (
            [0] * int(len(states) / 4)
            + [1] * int(len(states) / 4)
            + [2] * int(len(states) / 4)
            + [3] * int(len(states) / 4)
        )

        # Predict and record confusion matrix
        predictions = clf.predict(projections)
        for true_idx, true_feature_label in enumerate(SENTENCE_FEATURES):
            results[layer]["Four-Way"][true_feature_label] = {}

            for pred_idx, pred_feature_label in enumerate(SENTENCE_FEATURES):
                total = 0
                pred_total = 0
                for idx in range(len(labels)):
                    if labels[idx] == true_idx and predictions[idx] == pred_idx:
                        pred_total += 1
                    if labels[idx] == true_idx:
                        total += 1
                results[layer]["Four-Way"][true_feature_label][pred_feature_label] = (
                    pred_total / total
                )

    return results

def compute_overall_diff_vectors(model, splits):
    """Compute diff vectors over all pairs of datapoints in the 
    dataset.
    """

    torch.set_grad_enabled(False)
    # Store results
    diff_vectors = {}

    # Store all hidden states
    feature2states = {}

    ## Compute and cache all hidden states for each sentence feature
    print(f"Caching States")
    feature2states["probable"] = compute_hidden_states(
        model, splits, "probable"
    )
    feature2states["improbable"] = compute_hidden_states(
        model, splits, "improbable"
    )
    feature2states["impossible"] = compute_hidden_states(
        model, splits, "impossible"
    )
    feature2states["inconceivable"] = compute_hidden_states(
        model, splits, "inconceivable"
    )

    for sentence_feature in SENTENCE_FEATURES[1:]:
        diff_vectors[sentence_feature] = {}
        for layer in range(model.cfg.n_layers):
            mean_diff_vector = generate_mean_diff_vector(
                feature2states["probable"],
                feature2states[sentence_feature],
                layer,
            )
            diff_vectors[sentence_feature][layer] = mean_diff_vector
    return diff_vectors
    

def linear_analysis(model, train_splits, test_split):
    """Compute mean diff vectors for each feature, for each split.
    Assess how each mean diff vector partitions the relevant feature
    from probable examples (a 2-way classifier). Also, assess whether
    the mean diff vector partitions examples from all features (a
    4-way classifier).

    Finally, compute the mean diff vector over the entire dataset, and
    return it.
    """

    torch.set_grad_enabled(False)
    # Store results
    results_dict = {}

    # Store all hidden states
    train_feature2states = {}
    test_feature2states = {}

    ## Compute and cache all hidden states for each sentence feature
    print(f"Caching States")
    train_feature2states["probable"] = compute_hidden_states(
        model, train_splits, "probable"
    )
    train_feature2states["improbable"] = compute_hidden_states(
        model, train_splits, "improbable"
    )
    train_feature2states["impossible"] = compute_hidden_states(
        model, train_splits, "impossible"
    )
    train_feature2states["inconceivable"] = compute_hidden_states(
        model, train_splits, "inconceivable"
    )

    test_feature2states["probable"] = compute_hidden_states(
        model, [test_split], "probable"
    )
    test_feature2states["improbable"] = compute_hidden_states(
        model, [test_split], "improbable"
    )
    test_feature2states["impossible"] = compute_hidden_states(
        model, [test_split], "impossible"
    )
    test_feature2states["inconceivable"] = compute_hidden_states(
        model, [test_split], "inconceivable"
    )

    for sentence_feature in SENTENCE_FEATURES[1:]:
        # Compute mean diff vector between probable and each other feature
        print(f"Processing: {sentence_feature}")
        results_dict[sentence_feature] = {}

        layer2diff = {}
        for layer in range(model.cfg.n_layers):
            print(f"Layer: {layer}")
            mean_diff_vector = generate_mean_diff_vector(
                train_feature2states["probable"],
                train_feature2states[sentence_feature],
                layer,
            )
            layer2diff[layer] = mean_diff_vector

        # Featurize and Eval iterates over layers and tests all features,
        # returning a dictionary of results per layer
        print("Evaluating Diff Vector")
        test_results = eval_diff_vector(
            model, train_feature2states, test_feature2states, layer2diff
        )

        # Collect results
        for layer in range(model.cfg.n_layers):
            results_dict[sentence_feature][layer] = test_results[layer]

    return results_dict


def reformat_results(results):
    """Reformate results dictionaries into two dataframes"""
    two_way_results_df = {
        "vector_feature": [],
        "fold": [],
        "layer": [],
        "test_condition": [],
        "accuracy": [],
    }

    for split_idx, split_dict in enumerate(results):
        for train_feature in split_dict.keys():
            for layer in split_dict[train_feature].keys():
                for test_condition in split_dict[train_feature][layer]["Two-Way"].keys():
                    two_way_results_df["vector_feature"].append(train_feature)
                    two_way_results_df["fold"].append(split_idx)
                    two_way_results_df["layer"].append(layer)
                    two_way_results_df["test_condition"].append(test_condition)
                    two_way_results_df["accuracy"].append(
                        split_dict[train_feature][layer]["Two-Way"][test_condition]
                    )

    four_way_results_df = {
        "vector_feature": [],
        "fold": [],
        "layer": [],
        "test_feature": [],
        "predicted_feature": [],
        "accuracy": [],
    }

    for split_idx, split_dict in enumerate(results):
        for train_feature in split_dict.keys():
            for layer in split_dict[train_feature].keys():
                for test_feature in split_dict[train_feature][layer]["Four-Way"].keys():
                    for pred_feature in split_dict[train_feature][layer]["Four-Way"][
                        test_feature
                    ].keys():
                        four_way_results_df["vector_feature"].append(train_feature)
                        four_way_results_df["fold"].append(split_idx)
                        four_way_results_df["layer"].append(layer)
                        four_way_results_df["test_feature"].append(test_feature)
                        four_way_results_df["predicted_feature"].append(pred_feature)
                        four_way_results_df["accuracy"].append(
                            split_dict[train_feature][layer]["Four-Way"][test_feature][
                                pred_feature
                            ]
                        )

    return pd.DataFrame.from_dict(two_way_results_df), pd.DataFrame.from_dict(
        four_way_results_df
    )


def plot_two_way_analysis(results, figfolder, model_name):
    # For each layer, plot test/generalization acc for all different diff vectors
    eval_folder = os.path.join(figfolder, "Two_Way")
    os.makedirs(eval_folder, exist_ok=True)
    for layer in results["layer"].unique():
        layer_data = results[results["layer"] == layer]
        g = sns.catplot(
            data=layer_data,
            x="test_condition",
            y="accuracy",
            col="vector_feature",
            kind="bar",
            height=4,
            aspect=0.6,
        )
        g.set_xticklabels(rotation=45)
        g.set_axis_labels("", "Accuracy")
        g.set_titles("{col_name}")
        g.set(ylim=(0, 1))
        g.despine(left=True)
        # add overhead title
        g.fig.subplots_adjust(top=0.8)
        g.fig.suptitle(f"{model_name} Layer {layer} Evaluations")
        plt.savefig(
            os.path.join(eval_folder, f"layer_{layer}_eval.png"), bbox_inches="tight"
        )
        plt.close()
        plt.figure()


def plot_four_way_analysis(results, figfolder):
    # For each layer, plot confusion matrix for each vector
    eval_folder = os.path.join(figfolder, "Four_Way")
    os.makedirs(eval_folder, exist_ok=True)
    for layer in results["layer"].unique():
        for vector in results["vector_feature"].unique():
            layer_data = results[results["layer"] == layer]
            vector_data = layer_data[layer_data["vector_feature"] == vector]

            cells = np.zeros((4, 4))
            xlabs = []
            for i, test_feature in enumerate(results["test_feature"].unique()):
                xlabs.append(test_feature)
                ylabs = []
                for j, predicted_feature in enumerate(results["predicted_feature"].unique()):
                    cell_data = np.mean(
                        vector_data[
                            (vector_data["test_feature"]
                            == test_feature) & (vector_data["predicted_feature"]
                            == predicted_feature)
                        ]["accuracy"]
                    )
                    cells[i][j] = cell_data
                    ylabs.append(predicted_feature)

            g = sns.heatmap(
                data=cells,
                xticklabels=xlabs,
                yticklabels=ylabs,
                annot=True,
                cmap=sns.color_palette("light:#5A9", as_cmap=True),
            )
            plt.ylabel("Correct Class")
            plt.xlabel("Predicted Class")
            plt.title(f"{vector}: Layer {layer} Confusion Matrix")
            plt.savefig(
                os.path.join(eval_folder, f"layer_{layer}_{vector}.png"),
                bbox_inches="tight",
            )
            plt.close()
            plt.figure()


def reformat_diff_vectors(diff_vectors):
    """Diff vectors are ordered by sentence feature then layer, we want
    layer then sentence feature
    """
    reordered_diffs = []
    n_layers = len(diff_vectors["improbable"].keys())
    sentence_features = list(diff_vectors.keys())
    for layer in range(n_layers):
        diff_list = []
        label_list = []
        for sentence_feature in sentence_features:
            diff_list.append(
                diff_vectors[sentence_feature][layer].to(torch.float16).cpu()
            )
            label_list.append(sentence_feature)
        reordered_diffs.append(diff_list)
    return np.stack(reordered_diffs, axis=0), label_list


def plot_diff_vectors(diffs, label_list, figfolder, model_name):
    """For each layer, plot the RSM of all computed mean diff vectors"""
    rsm_folder = os.path.join(figfolder, "RSM")
    os.makedirs(rsm_folder, exist_ok=True)
    for i in range(len(diffs)):

        similarities = cosine_similarity(diffs[i])
        plt.imshow(similarities)
        plt.colorbar()
        plt.xticks(
            range(len(similarities)), labels=label_list, rotation=90, fontsize=10
        )
        plt.yticks(range(len(similarities)), labels=label_list, fontsize=10)

        plt.title(f"{model_name} Layer {i} Mean-Diff RSM")
        plt.savefig(os.path.join(rsm_folder, f"layer_{i}_rsm.png"), bbox_inches="tight")
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
    dataset = SentenceFeaturesDataset(file="stimuli_with_syntax")
    fold_proportion = 1.0 / args.k_folds
    splits = random_split(dataset, [fold_proportion] * args.k_folds)

    k_fold_results = []
    for i in range(args.k_folds):
        print(f"Fold {i}\n++++++++++++")
        test_split = splits[i]
        train_idxs = list(range(args.k_folds))
        train_idxs.remove(i)
        train_splits = [splits[idx] for idx in train_idxs]

        results = linear_analysis(model, train_splits, test_split)
        k_fold_results.append(results)

    two_way_results, four_way_results = reformat_results(k_fold_results)
    two_way_results.to_csv(os.path.join(outfolder, "two_way_results.csv"))
    four_way_results.to_csv(os.path.join(outfolder, "four_way_results.csv"))

    # Plot results
    plot_two_way_analysis(two_way_results, figfolder, args.model)
    plot_four_way_analysis(four_way_results, figfolder)

    # Diff vectors are computed over the entire corpus
    diff_vectors = compute_overall_diff_vectors(model, splits)
    pkl.dump(diff_vectors, open(os.path.join(outfolder, "diff_vectors.pkl"), "wb"))

    # Plot RSM of Diff Vectors
    diff_vectors, label_list = reformat_diff_vectors(diff_vectors)
    plot_diff_vectors(diff_vectors, label_list, figfolder, args.model)

    # Plot projections on all diff vectors
    plot_projections(diff_vectors, splits)
