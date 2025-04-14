"""
This file is used to generate the difference vectors
that support the remaining analyses in the study.
"""

import argparse
import os
from collections import defaultdict
import pickle as pkl

import pandas as pd
import numpy as np

import torch
import transformer_lens
import transformer_lens.utils as utils
from tqdm import tqdm

from Utilities.data import SentenceFeaturesDataset

SENTENCE_FEATURES = [
    ("probable", "improbable"),
    ("probable", "impossible"),
    ("probable", "inconceivable"),
    ("improbable", "impossible"),
    ("improbable", "inconceivable"),
    ("impossible", "inconceivable"),
]


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
        default="./Study_1/results/",
        type=str,
        help="Folder to put results",
    )

    parser.add_argument(
        "-s", "--splits", default=5, type=int, help="Number of CV folds + 1 Eval Fold"
    )

    args = parser.parse_args()
    return args


def compute_hidden_states(model, dataset, feature_label):
    """Extract the hidden states of the final token (".")
    from every sentence exhibiting a particular feature from
    every split.
    """
    layer2states = defaultdict(list)

    for datum in tqdm(dataset, desc=f"Computing {feature_label} States"):
        prompt = datum[feature_label]
        tokens = model.to_tokens(prompt, prepend_bos=True)
        _, cache = model.run_with_cache(tokens)

        for layer in range(model.cfg.n_layers):
            # Extract the final hidden state, corresponding to the "." token
            layer2states[layer].append(
                cache[utils.get_act_name("resid_post", layer)].cpu()[0, -1]
            )
    # Convert to a numpy array of states for indexing
    layer2states = {k: torch.stack(v, dim=0) for k, v in layer2states.items()}
    return layer2states


def generate_mean_diff_vector(feature0_states, feature1_states):
    """Compute the diff vector hidden states of sentences exhibiting two features"""
    diffs = [
        feature1_states[i] - feature0_states[i] for i in range(len(feature0_states))
    ]
    diffs = torch.stack(diffs, dim=0)
    mean_diff = torch.mean(diffs, dim=0)
    return mean_diff


def eval_diff_vector(feature2state, sentence_features, layer2diffs, eval_split):
    """Given mean diff vectors corresponding to a pair of sentence features,
    see how well the diff vectors classify pairs of held out sentences
    """

    results = {}

    for layer in layer2diffs.keys():
        diff_vector = layer2diffs[layer]

        layer_accs = []
        for eval_pair in eval_split:
            low_state = feature2state[sentence_features[0]][layer][eval_pair]
            high_state = feature2state[sentence_features[1]][layer][eval_pair]

            low_projection = torch.dot(low_state, diff_vector)
            high_projection = torch.dot(high_state, diff_vector)

            if high_projection > low_projection:
                layer_accs.append(1)
            else:
                layer_accs.append(0)
        results[layer] = np.mean(layer_accs)

    return results


def compute_overall_diff_vectors(feature2states, best_layer, outfolder):
    """Compute diff vectors over all pairs of datapoints in the
    dataset.
    """
    torch.set_grad_enabled(False)
    diff_vectors = {}
    for sentence_features in SENTENCE_FEATURES:
        diff_vectors["_".join(sentence_features)] = generate_mean_diff_vector(
            feature2states[sentence_features[0]][best_layer],
            feature2states[sentence_features[1]][best_layer],
        )
    pkl.dump(diff_vectors, open(os.path.join(outfolder, f"diff_vectors_layer_{best_layer}.pkl"), "wb"))


def run_split(model, feature2states, train_splits, eval_split, layer=None):
    """Given a split, compute the mean difference vectors for each
    feature pair. Assess how each of these difference vectors
    classify their respective sentence pairs in the eval split.

    Repeat this for each layer, report classification accuracy
    for each feature pair.
    """

    torch.set_grad_enabled(False)
    if layer is None:
        layers = range(model.cfg.n_layers)
    else:
        layers = [layer]

    # Store results
    results_dict = {}

    train_splits = train_splits.reshape(-1)
    for sentence_features in SENTENCE_FEATURES:

        # Compute mean diff vector between two sentence features
        print(f"Processing: {sentence_features}")
        results_dict["_".join(sentence_features)] = {}

        layer2diff = {}
        for layer in layers:
            print(f"Layer: {layer}")

            mean_diff_vector = generate_mean_diff_vector(
                feature2states[sentence_features[0]][layer][train_splits],
                feature2states[sentence_features[1]][layer][train_splits],
            )
            layer2diff[layer] = mean_diff_vector

        # Evaluate how well this difference vector classifies stimuli
        eval_results = eval_diff_vector(
            feature2states, sentence_features, layer2diff, eval_split
        )
        results_dict["_".join(sentence_features)] = eval_results

    return results_dict


def run_probability_eval(model, dataset, test_split):
    """Evaluate pairs of sentences based on their relative probability
    (equivalently, their relative loss under a model)
    """
    results = {}
    for sentence_features in SENTENCE_FEATURES:
        print(f"Probability Eval: {sentence_features}")
        features_acc = []
        for set_idx in test_split:
            datum = dataset[set_idx]
            low_tokens = model.to_tokens(datum[sentence_features[0]], prepend_bos=True)
            high_tokens = model.to_tokens(datum[sentence_features[1]], prepend_bos=True)

            low_loss = model(low_tokens, return_type="loss")
            high_loss = model(high_tokens, return_type="loss")

            if high_loss > low_loss:
                features_acc.append(1)
            else:
                features_acc.append(0)

        results["_".join(sentence_features)] = np.mean(features_acc).item()
    return results


def process_cv_results(cv_results, outfolder):
    """Format CV results into a dataframe and save them.
    Additionally, find the layer that had the best average accuracy
    across all diff vectors, and report it.
    """
    df = {
        "fold": [],
        "diff_vector": [],
        "layer": [],
        "accuracy": [],
    }

    for fold, fold_results in enumerate(cv_results):
        for diff_vector, vector_results in fold_results.items():
            for layer, acc in vector_results.items():
                df["fold"].append(fold)
                df["diff_vector"].append(diff_vector)
                df["layer"].append(layer)
                df["accuracy"].append(acc.item())
    df = pd.DataFrame.from_dict(df)
    df.to_csv(os.path.join(outfolder, "cv_results.csv"), index=False)

    layer_df = df.groupby("layer")
    best_layer = -1
    best_acc = 0.0
    for layer, grp in layer_df:
        avg_acc = grp["accuracy"].mean()
        if avg_acc > best_acc:
            best_layer = layer
            best_acc = avg_acc
    print(f"Best Layer: {best_layer}, Best Acc: {best_acc}")
    return best_layer


def process_test_results(diff_vector_results, probability_results, outfolder):
    """Format and record results from diff vectors and output probability"""
    df = {
        "method": [],
        "features": [],
        "accuracy": [],
    }

    for features, results in diff_vector_results.items():
        for _, acc in results.items():
            df["method"].append("Difference Vector")
            df["features"].append(features)
            df["accuracy"].append(acc)

    for features, acc in probability_results.items():
        df["method"].append("Probability")
        df["features"].append(features)
        df["accuracy"].append(acc)

    pd.DataFrame.from_dict(df).to_csv(os.path.join(outfolder, "test_results.csv"), index=False)


if __name__ == "__main__":
    # Parse Args
    args = parse_arguments()

    ### Set Up Output
    outfolder = os.path.join(args.outfolder, args.model)
    os.makedirs(outfolder, exist_ok=True)

    ### Set up model
    torch.set_grad_enabled(False)
    torch.manual_seed(19)
    np.random.seed(19)

    model = transformer_lens.HookedTransformer.from_pretrained(
        args.model,
        device="cuda",
        dtype="bfloat16",
    )

    ### Load Dataset and Cache hidden states
    dataset = SentenceFeaturesDataset(file="../data/shades_of_zero")

    # Store all hidden states
    feature2states = {}

    # Cache all hidden states for each sentence feature
    print(f"Caching States")
    feature2states["probable"] = compute_hidden_states(model, dataset, "probable")
    feature2states["improbable"] = compute_hidden_states(model, dataset, "improbable")
    feature2states["impossible"] = compute_hidden_states(model, dataset, "impossible")
    feature2states["inconceivable"] = compute_hidden_states(
        model, dataset, "inconceivable"
    )

    ### Split Data and Iterate
    indices = torch.randperm(len(dataset)).numpy()
    split_indices = np.array_split(indices, args.splits)

    cv_splits = split_indices[:-1]
    test_split = split_indices[-1]

    cv_results = []
    k_folds = args.splits - 1
    for i in range(k_folds):
        print(f"Fold {i}\n++++++++++++")
        eval_split = cv_splits[i]
        train_idxs = list(range(k_folds))
        train_idxs.remove(i)
        train_splits = np.array([cv_splits[idx] for idx in train_idxs])

        results = run_split(model, feature2states, train_splits, eval_split)
        cv_results.append(results)

    best_layer = process_cv_results(cv_results, outfolder)

    # Evaluate performance on a held out test set
    test_results = run_split(
        model, feature2states, np.array(cv_splits), test_split, layer=best_layer
    )
    probability_results = run_probability_eval(model, dataset, test_split)

    process_test_results(test_results, probability_results, outfolder)
    compute_overall_diff_vectors(feature2states, best_layer, outfolder)
