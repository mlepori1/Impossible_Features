"""
This file projects stimuli on to difference vectors, and
also collects overall sentence probabiliity (in the form of loss).

Data are assumed to be csv files that contains sentences to project
and an optional class label
"""

import argparse
import os
from collections import defaultdict
import pickle as pkl

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

import torch
import transformer_lens
import transformer_lens.utils as utils
from tqdm import tqdm

VECTORS = [
    "probable_improbable",
    "improbable_impossible",
    "impossible_inconceivable",
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
        default="./Study_2/results/",
        type=str,
        help="Folder to put results",
    )

    parser.add_argument(
        "--dataset",
        default="../data/shades_reformated.csv",
        type=str,
        help="Data file to process",
    )

    parser.add_argument(
        "--diff_vectors",
        default="./Study_1/results/gemma-2-2b/diff_vectors_layer_11.pkl",
        type=str,
        help="Location of pickled diff vectors",
    )

    parser.add_argument(
        "--pca",
        default="./Study_1/results/gemma-2-2b/pca.pkl",
        type=str,
        help="Location of pickled PCA",
    )

    args = parser.parse_args()
    return args


def project_vectors(model, dataset, diff_vectors, pca, random_vectors, layer):
    """Projects sentence pairs along diff vectors and assesses
    accuracy
    """
    results = {
        "probable_improbable": [],
        "improbable_impossible": [],
        "impossible_inconceivable": [],
        "pca_0": [],
        "pca_1": [],
        "pca_2": [],
        "random_0": [],
        "random_1": [],
        "random_2": [],
    }

    for _, row in tqdm(
        dataset.iterrows(), total=len(dataset), desc="Difference Vector Projection"
    ):
        # Process sentences and extract hidden states
        sentence = row["sentence"]
        tok = model.to_tokens(sentence, prepend_bos=True)
        _, cache_0 = model.run_with_cache(tok)
        state = cache_0[utils.get_act_name("resid_post", layer)].cpu()[0, -1]

        # Get the diff vector and project
        for vector in VECTORS:
            diff_vector = diff_vectors[vector]
            projection = torch.dot(state, diff_vector).item()
            results[vector].append(projection)
        
        # Project state on to pca vectors
        pca_projs = pca.transform(state.float().numpy().reshape(1, -1))
        for pca_idx, proj in enumerate(pca_projs[0]):
            results[f"pca_{pca_idx}"].append(proj)

        # Project state on random vectors
        for rand_idx, rand_vector in enumerate(random_vectors):
            projection = torch.dot(state, rand_vector).item()
            results[f"random_{rand_idx}"].append(projection)

    return results


def get_loss(model, dataset):
    """Get loss for each sentence in the dataset"""
    results = []

    for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Loss computation"):
        # Process sentences
        sentence = row["sentence"]
        tok = model.to_tokens(sentence, prepend_bos=True)
        loss = model(tok, return_type="loss").item()
        results.append(loss)
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

    ### Load PCA Transformation
    pca = pkl.load(open(args.pca, "rb"))

    ### Load up 3 random vectors
    random_vectors = torch.randn(3, model.cfg.d_model, generator=torch.Generator().manual_seed(19)).bfloat16()

    ### Run Diff Vector Projections
    projections = project_vectors(model, dataset, diff_vectors, pca, random_vectors, layer)

    ### Get losses
    losses = get_loss(model, dataset)

    # Save off results
    for vector in VECTORS:
        dataset[vector] = projections[vector]

    for control_proj in ["pca_0", "pca_1", "pca_2", "random_0", "random_1", "random_2"]:
        dataset[control_proj] = projections[control_proj]

    dataset["loss"] = losses

    dataname = args.dataset.split("/")[-1]
    dataset.to_csv(os.path.join(outfolder, f"projected_{dataname}"), index=False)
