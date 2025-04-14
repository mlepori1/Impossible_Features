"""
This file classifies new data according to both projections
along difference vectors and output probability.

This file assumes the following dataset columns:

"sentence_0": This sentence should be the more "normal" sentence
            in the pair
"sentence_1": This sentence should be the less "normal" sentence
"modal_0": The modal feature associated with sentence 0
"modal_1": The modal feature associated with sentence 1

All other columns are used as metadata to futher define the kinds of
pairs that are being compared 
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
        "--dataset",
        default="../data/sherlock.csv",
        type=str,
        help="Data file to process",
    )

    parser.add_argument(
        "--diff_vectors",
        default="./Study_1/results/gemma-2-2b/diff_vectors_layer_11.pkl",
        type=str,
        help="Location of pickled diff vectors",
    )

    args = parser.parse_args()
    return args



def process_test_results(diff_vector_results, probability_results, outfolder, data_name):
    """Format and record results from diff vectors and output probability"""
    df = {
        "method": [],
        "pair_type": [],
        "accuracy": [],
    }

    for pair_type, acc in diff_vector_results.items():
        df["method"].append("Difference Vector")
        df["pair_type"].append(pair_type)
        df["accuracy"].append(acc)

    for pair_type, acc in probability_results.items():
        df["method"].append("Probability")
        df["pair_type"].append(pair_type)
        df["accuracy"].append(acc)

    pd.DataFrame.from_dict(df).to_csv(
        os.path.join(outfolder, f"results_{data_name}"), index=False
    )


def run_diff_vectors(model, dataset, diff_vectors, layer):
    """Projects sentence pairs along diff vectors and assesses
    accuracy
    """

    results = defaultdict(list)

    for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Difference Vector Eval"):
        # Extract pair type
        col_names = row.index.tolist()
        col_names = [
            col for col in col_names if col not in ["sentence_0", "sentence_1"]
        ]
        pair_type = "_".join([row[col] for col in col_names])

        # Process sentences and extract hidden states
        sent0 = row["sentence_0"]
        sent1 = row["sentence_1"]

        tok_0 = model.to_tokens(sent0, prepend_bos=True)
        _, cache_0 = model.run_with_cache(tok_0)
        state_0 = cache_0[utils.get_act_name("resid_post", layer)].cpu()[0, -1]

        tok_1 = model.to_tokens(sent1, prepend_bos=True)
        _, cache_1 = model.run_with_cache(tok_1)
        state_1 = cache_1[utils.get_act_name("resid_post", layer)].cpu()[0, -1]

        # Get the correct diff vector
        diff_vector_name = row["modal_0"] + "_" + row["modal_1"]
        diff_vector = diff_vectors[diff_vector_name]

        # Project and compute accuracy
        low_projection = torch.dot(state_0, diff_vector)
        high_projection = torch.dot(state_1, diff_vector)

        if high_projection > low_projection:
            results[pair_type].append(1)
        else:
            results[pair_type].append(0)

    results = {k: np.mean(v).item() for k, v in results.items()}
    return results

def run_probability(model, dataset):
    """Evaluate pairs of sentences based on their relative probability
    (equivalently, their relative loss under a model)
    """
    results = defaultdict(list)

    for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Probability eval"):
        # Extract pair type
        col_names = row.index.tolist()
        col_names = [
            col for col in col_names if col not in ["sentence_0", "sentence_1"]
        ]
        pair_type = "_".join([row[col] for col in col_names])

        # Process sentences and extract hidden states
        sent0 = row["sentence_0"]
        sent1 = row["sentence_1"]

        tok_0 = model.to_tokens(sent0, prepend_bos=True)
        loss_0 = model(tok_0, return_type="loss")

        tok_1 = model.to_tokens(sent1, prepend_bos=True)
        loss_1 = model(tok_1, return_type="loss")

        if loss_1 > loss_0:
            results[pair_type].append(1)
        else:
            results[pair_type].append(0)

    results = {k: np.mean(v).item() for k, v in results.items()}
    return results

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

    ### Load Dataset
    dataset = pd.read_csv(args.dataset)

    ### Load Diff Vectors
    layer = int(args.diff_vectors.split("_")[-1][:-4])  # Layer is in filename
    diff_vectors = pkl.load(open(args.diff_vectors, "rb"))

    ### Run Diff Vector Evaluation
    diff_results = run_diff_vectors(model, dataset, diff_vectors, layer)

    ### Run Output Evaluation
    prob_results = run_probability(model, dataset)

    # Save off results
    process_test_results(diff_results, prob_results, outfolder, args.dataset.split("/")[-1])
