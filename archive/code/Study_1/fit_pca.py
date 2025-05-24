"""
This file fits and saves a SKlearn PCA module to wikitext data,
in order to serve as a baseline for all evaluations 
"""

import argparse
import os
import pickle as pkl

from datasets import load_dataset

import numpy as np

import torch
import transformer_lens
import transformer_lens.utils as utils

from sklearn.decomposition import PCA

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
        default="./Study_2/results/",
        type=str,
        help="Folder to put results",
    )

    parser.add_argument(
        "--diff_vectors",
        default="./Study_1/results/gemma-2-2b/diff_vectors_layer_11.pkl",
        type=str,
        help="Location of pickled diff vectors",
    )

    args = parser.parse_args()
    return args


def process_dataset(model, layer):
    """Projects sentence pairs along diff vectors and assesses
    accuracy
    """
    # Load the WikiText-103 dataset
    dataset = load_dataset("wikitext", "wikitext-103-v1")["validation"]
    states = []
    for row in tqdm(
        dataset, total=len(dataset), desc="Processing wikitext"
    ):
        # Process sentences and extract hidden states of the last token
        sentence = row["text"]
        tok = model.to_tokens(sentence, prepend_bos=True)
        _, cache = model.run_with_cache(tok)
        state = cache[utils.get_act_name("resid_post", layer)].cpu().float().numpy()[0, -1]
        states.append(state)

    return states

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

    ### Get Layer from Diff Vectors
    layer = int(args.diff_vectors.split("_")[-1][:-4])  # Layer is in filename

    states = process_dataset(model, layer)

    # Fit PCA
    pca = PCA(n_components=3)
    pca = pca.fit(np.stack(states, axis=0))

    pkl.dump(pca, open(os.path.join(outfolder, "pca.pkl"), "wb"))
