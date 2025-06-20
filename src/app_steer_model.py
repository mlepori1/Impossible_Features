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
import numpy as np

import torch
import transformer_lens

import utils


HF_2_TL = {
    "google/gemma-2-2b": "gemma-2-2b",
    "meta-llama/Llama-3.2-3B": "meta-llama/Llama-3.2-3B",
    }


def steering_vector_hook(
    resid,
    hook,
    vector,
    multiplier,
) -> torch.Tensor:

    vector = torch.Tensor(vector).to("cuda").reshape(1, 1, -1) * multiplier
    vector = vector.repeat(1, resid.shape[1], 1)
    resid += vector
    return resid


def get_vectors(model):

    vector_paths = {}
    vector_path = f"../artifacts/{model}/Linear_Representation/Linear_Representation/"

    for filename in os.listdir(vector_path):
        if "probable_improbable" in filename:
            layer = int(filename.split("_")[-1][:-4])
            vector_paths["improbable"] = (os.path.join(vector_path, filename), layer)
        if "probable_impossible" in filename:
            layer = int(filename.split("_")[-1][:-4])
            vector_paths["impossible"] = (os.path.join(vector_path, filename), layer)
        if "probable_inconceivable" in filename:
            layer = int(filename.split("_")[-1][:-4])
            vector_paths["inconceivable"] = (os.path.join(vector_path, filename), layer)

    return vector_paths

def compute_stop_probability(model, logits):
    """Compute the probability of the period token after a generation.
    Used to determine when to stop generating.
    """
    period_token = model.to_tokens(".", prepend_bos=False)
    return torch.softmax(logits[0, -1], dim=0)[period_token].float().cpu()


if __name__ == "__main__":
    # Parse Args
    config = utils.get_config()

    ### Set Up Output
    model_name = config["model"]

    outfolder = os.path.join("../results/", model_name, "Steering")
    os.makedirs(outfolder, exist_ok=True)

    ### Set up model
    torch.set_grad_enabled(False)
    model = transformer_lens.HookedTransformer.from_pretrained(
        HF_2_TL[model_name],
        device="cuda",
        dtype="bfloat16",
    )

    vector_dict = get_vectors(model_name)
    prefixes = pd.read_csv("../data/steering/prefixes.csv")

    ### Set up data
    df = {
        "prefix": [],
        "steering_concept": [],
        "multiplier": [],
        "rank": [],
        "full_generation": [],
        "first_token_generation": [],
        "surprisal": [],
    }

    for prefix in tqdm(prefixes["prefix"]):
        ## First compute generations with no intervention

        # First, generate top 5 next token predictions
        logits = model(prefix, prepend_bos=True, return_type="logits")
        top_k_indices = torch.topk(logits[0, -1], 5).indices

        # Compute the surprisals of these tokens
        log_probs = torch.log_softmax(logits[0, -1], dim=0)
        surprisals = [-1 * log_probs[idx].item() for idx in top_k_indices]
        df["surprisal"] += surprisals

        seed_generations = model.to_string(top_k_indices.reshape(-1, 1))

        # For each seed generation, generate up to 4 more tokens
        for gen_idx, seed_generation in enumerate(seed_generations):
            further_generations = []
            prob_stop = []
            for _ in range(4):
            
                logits = model(
                    prefix + seed_generation + "".join(further_generations),
                    return_type="logits",
                )
                prob_stop.append(compute_stop_probability(model, logits))
                top_index = torch.topk(logits[0, -1], 1).indices
                further_generations.append(model.to_string(top_index.reshape(-1, 1))[0])
            
            # Return full generation up until the point where
            # the model was most likely to produce a period token.
            full_generation = seed_generation + "".join(further_generations[:np.argmax(prob_stop)])

            # Store data
            df["prefix"].append(prefix)   
            df["steering_concept"].append("None")
            df["multiplier"].append(None)
            df["rank"].append(gen_idx)
            df["full_generation"].append(full_generation)
            df["first_token_generation"].append(seed_generation)

        ## Now, repeat this, except with steering
        for concept_name, (vector_path, layer) in vector_dict.items():
        
            steering_vector = pkl.load(
                open(vector_path, "rb")
            )

            for multiplier in [3, 5]:
                # First, generate top 5 next token predictions

                # Run once without hooks, to calculate surprisal later
                plain_logits = model(prefix, prepend_bos=True, return_type="logits")

                # Now run with steering
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
                top_k_indices = torch.topk(logits[0, -1], 5).indices

                # Compute the surprisals of these tokens, according to the 
                # no-intervention run
                log_probs = torch.log_softmax(plain_logits[0, -1], dim=0)
                surprisals = [-1 * log_probs[idx].item() for idx in top_k_indices]
                df["surprisal"] += surprisals

                seed_generations = model.to_string(top_k_indices.reshape(-1, 1))

                # For each seed generation, generate up to 4 more tokens
                for gen_idx, seed_generation in enumerate(seed_generations):
                    further_generations = []
                    prob_stop = []
                    for _ in range(4):
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
                            prefix + seed_generation + "".join(further_generations), 
                            return_type="logits", 
                            fwd_hooks=fwd_hooks
                        )
                        prob_stop.append(compute_stop_probability(model, logits))
                        top_index = torch.topk(logits[0, -1], 1).indices
                        further_generations.append(model.to_string(top_index.reshape(-1, 1))[0])
                    
                    # Return full generation up until the point where
                    # the model was most likely to produce a period token.
                    full_generation = seed_generation + "".join(further_generations[:np.argmax(prob_stop)])

                    # Store data
                    df["prefix"].append(prefix)   
                    df["steering_concept"].append(concept_name)
                    df["multiplier"].append(multiplier)
                    df["rank"].append(gen_idx)
                    df["full_generation"].append(full_generation)
                    df["first_token_generation"].append(seed_generation)

    pd.DataFrame.from_dict(df).to_csv(
        os.path.join(outfolder, f"steering_generations.csv"), index=False
    )
