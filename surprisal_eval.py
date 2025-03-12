"""
This file runs an LM over prompts exhibiting different
sentence-level features, and computes the surprisal of the
sentences. Then, it creates a graph similar to Fig 3 of 
Shades of Zero
"""

import argparse
import os
import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import transformer_lens
import torch.nn.functional as F

from data import SentenceFeaturesDataset


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        default="gemma-2-2b",
        help="Model to use as backbone for analysis",
    )

    parser.add_argument(
        "-s",
        "--suffix",
        action="store_true",
        help="Whether to just compute the surprisal on the suffix"
    )

    parser.add_argument(
        "--outfolder",
        default="./results/surprisal",
        type=str,
        help="Folder to put results",
    )
 
    args = parser.parse_args()
    return args


def args2dict(args):
    return {
        "model": args.model,
    }


def process_prompt(model, prompt, base, suffix=True):
    # Compute average surprisal over the continuation
    tokens = model.to_tokens(prompt, prepend_bos=True)

    if suffix:
        prefix_len = len(model.to_tokens(base, prepend_bos=True)[0])
        labels = tokens[0, prefix_len:]
        logits =  model(tokens, return_type="logits")[0, prefix_len-1:-1]
    else:
        labels = tokens[0, 1:]
        logits =  model(tokens, return_type="logits")[0, :-1]

    logprobs = F.log_softmax(logits, dim=-1)
    logprobs = logprobs[range(len(labels)), labels]
    return torch.mean(-1 * logprobs).item()


def surprisal_eval(model, dataset, just_suffix=False):
    """Run model, get surprisal for all stimuli"""
    torch.set_grad_enabled(False)

    probable_surprisal = []
    improbable_surprisal = []
    impossible_surprisal = []
    inconceivable_surprisal = []
    syntactic_surprisal = []
    shuffled_surprisal = []


    total_samples = len(dataset)
    for i in range(total_samples):

        if i % 10 == 0:
            print(f"Progress: {round(i/total_samples, 3)}")

        if just_suffix:
            probable_surprisal.append(process_prompt(model, dataset[i]["probable"], dataset[i]["base"]))
            improbable_surprisal.append(process_prompt(model, dataset[i]["improbable"], dataset[i]["base"]))
            impossible_surprisal.append(process_prompt(model, dataset[i]["impossible"], dataset[i]["base"]))
            inconceivable_surprisal.append(process_prompt(model, dataset[i]["inconceivable"], dataset[i]["base"]))
            syntactic_surprisal.append(process_prompt(model, dataset[i]["syntactic"], dataset[i]["base"]))
            shuffled_surprisal.append(process_prompt(model, dataset[i]["shuffled"], dataset[i]["base"]))

        else:
            probable_surprisal.append(process_prompt(model, dataset[i]["probable"], dataset[i]["base"], suffix=False))
            improbable_surprisal.append(process_prompt(model, dataset[i]["improbable"], dataset[i]["base"], suffix=False))
            impossible_surprisal.append(process_prompt(model, dataset[i]["impossible"], dataset[i]["base"], suffix=False))
            inconceivable_surprisal.append(process_prompt(model, dataset[i]["inconceivable"], dataset[i]["base"], suffix=False))
            syntactic_surprisal.append(process_prompt(model, dataset[i]["syntactic"], dataset[i]["base"], suffix=False))
            shuffled_surprisal.append(process_prompt(model, dataset[i]["shuffled"], dataset[i]["base"], suffix=False))


    return {
        "probable": probable_surprisal,
        "improbable": improbable_surprisal,
        "impossible": impossible_surprisal,
        "inconceivable": inconceivable_surprisal,
        "syntactic": syntactic_surprisal,
        "shuffled": shuffled_surprisal,
    }

def plot_surprisal(results, figfolder, model_name, suffix=True):

    n_per_condition = len(results["probable"])
    condition = ["probable"] * n_per_condition +\
    ["improbable"] * n_per_condition +\
    ["impossible"] * n_per_condition +\
    ["inconceivable"] * n_per_condition +\
    ["syntactic"] * n_per_condition +\
    ["shuffled"] * n_per_condition

    surprisal = results["probable"] + results["improbable"] + results["impossible"] +\
        results["inconceivable"] + results["syntactic"] + results["shuffled"]
    
    df = pd.DataFrame.from_dict({"condition": condition, "surprisal": surprisal})
    sns.barplot(df, x="condition", y="surprisal")
    if suffix:
        plt.title(f"{model_name}: Suffix Surprisal by Condition")
        plt.savefig(os.path.join(figfolder, "surprisal_suffix.png"))
    else:
        plt.title(f"{model_name}: Surprisal by Condition")
        plt.savefig(os.path.join(figfolder, "surprisal.png"))    

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
    dataset = SentenceFeaturesDataset()
    results = surprisal_eval(model, dataset, just_suffix=args.suffix)

    # Plot results
    plot_surprisal(results, figfolder, args.model, suffix=args.suffix)
