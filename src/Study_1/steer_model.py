"""
This file uses the difference vectors to steer
the generations of novel prefixes in order to
assess the causal effect of modal difference vectors
"""

import argparse
import os
import pickle as pkl
from functools import partial

from tqdm import tqdm
import pandas as pd

import torch
import transformer_lens


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
        help="File to put results",
    )

    parser.add_argument(
        "--dataset",
        default="../data/study_1_generated_prefixes.txt",
        type=str,
        help="Novel prefixes to process",
    )

    parser.add_argument(
        "--diff_vectors",
        default="./Study_1/results/gemma-2-2b/diff_vectors_layer_11.pkl",
        type=str,
        help="Location of pickled diff vectors",
    )

    args = parser.parse_args()
    return args


def steering_vector_hook(
    resid,
    hook,
    vector,
    multiplier,
) -> torch.Tensor:

    vector = vector.to("cuda").reshape(1, 1, -1) * multiplier
    vector = vector.repeat(1, resid.shape[1], 1)
    resid += vector
    return resid


def extract_prefixes(datafile):
    f = open(datafile, "r")
    prefixes = []
    for line in f.readlines():
        prefix = line.split(".")[-1].strip()
        prefixes.append(prefix)
    return prefixes[:10]


if __name__ == "__main__":
    # Parse Args
    args = parse_arguments()
    outfolder = os.path.join(args.outfolder, args.model)
    ### Set up model
    torch.set_grad_enabled(False)
    model = transformer_lens.HookedTransformer.from_pretrained(
        args.model,
        device="cuda",
        dtype="bfloat16",
    )

    ### Get diff vectors
    layer = int(args.diff_vectors.split("_")[-1][:-4])  # Layer is in filename
    steering_vectors = pkl.load(open(args.diff_vectors, "rb"))

    SENTENCE_FEATURES = [
        ("probable", "improbable"),
        ("probable", "impossible"),
        ("probable", "inconceivable"),
    ]

    ### Set up data
    prefixes = extract_prefixes(args.dataset)

    df = {
        "prefix": [],
        "steering_concept": [],
        "multiplier": [],
        "rank": [],
        "generation": [],
    }

    ### Run steering and generations
    for prefix in tqdm(prefixes):

        # First collect baseline generations
        logits = model(prefix, return_type="logits")
        top_k_indices = torch.topk(logits[0, -1], 20).indices
        generations = model.to_string(top_k_indices.reshape(-1, 1))

        for rank, generation in enumerate(generations):
            df["prefix"].append(prefix)
            df["steering_concept"].append("None")
            df["multiplier"].append(0)
            df["rank"].append(rank)
            df["generation"].append(generation)

        # Next apply steering and collect generations
        for features in SENTENCE_FEATURES:

            # Get the correct diff vector
            vector_name = features[0] + "_" + features[1]
            steering_vector = steering_vectors[vector_name]

            for multiplier in [5]:
                fwd_hooks = []
                fwd_hooks.append(
                    (
                        f"blocks.{layer}.hook_resid_post",
                        partial(
                            steering_vector_hook,
                            vector=steering_vector,
                            multiplier=multiplier,
                        ),
                    )
                )
                logits = model.run_with_hooks(
                    prefix, return_type="logits", fwd_hooks=fwd_hooks
                )

                top_k_indices = torch.topk(logits[0, -1], 10).indices
                generations = model.to_string(top_k_indices.reshape(-1, 1))

                # Generate one more token
                final_generations = []
                for generation in generations:
                    fwd_hooks = []
                    fwd_hooks.append(
                        (
                            f"blocks.{layer}.hook_resid_post",
                            partial(
                                steering_vector_hook,
                                vector=steering_vector,
                                multiplier=multiplier,
                            ),
                        )
                    )
                    logits = model.run_with_hooks(
                        prefix + generation, return_type="logits", fwd_hooks=fwd_hooks
                    )

                    top_k_indices = torch.topk(logits[0, -1], 1).indices
                    new_gens = model.to_string(top_k_indices.reshape(-1, 1))
                    final_generations.append(generation + new_gens[0])

                for rank, generation in enumerate(final_generations):
                    df["prefix"].append(prefix)
                    df["steering_concept"].append(features[1])
                    df["multiplier"].append(multiplier)
                    df["rank"].append(rank)
                    df["generation"].append(generation)

    pd.DataFrame.from_dict(df).to_csv(os.path.join(outfolder, "steering_generations_2.csv"), index=False)
