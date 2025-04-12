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

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix

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

class TieBreakerKNN(KNeighborsClassifier):

    def _get_neighbors(self, distances, indices, k):
        selected_indices = []
        for i in range(distances.shape[0]):
            unique_distances = np.unique(distances[i])
            selected = []
            for dist in unique_distances:
                tied_points = indices[i][distances[i] == dist]
                np.random.shuffle(tied_points)
                remainder = k - len(selected)
                selected.extend(tied_points[:remainder])

                if len(selected) == k:
                    break
            selected_indices.append(selected)
        return np.array(selected_indices)

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        all_neighbors = self._fit_X.shape[0]
        distances, indices = super().kneighbors(
            X, n_neighbors=all_neighbors, return_distance=True
        )
        indices = self._get_neighbors(distances, indices, self.n_neighbors)
        return (distances, indices) if return_distance else indices

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


    total_samples = len(dataset)
    for i in range(total_samples):

        if i % 10 == 0:
            print(f"Progress: {round(i/total_samples, 3)}")

        if just_suffix:
            probable_surprisal.append(process_prompt(model, dataset[i]["probable"], dataset[i]["base"]))
            improbable_surprisal.append(process_prompt(model, dataset[i]["improbable"], dataset[i]["base"]))
            impossible_surprisal.append(process_prompt(model, dataset[i]["impossible"], dataset[i]["base"]))
            inconceivable_surprisal.append(process_prompt(model, dataset[i]["inconceivable"], dataset[i]["base"]))

        else:
            probable_surprisal.append(process_prompt(model, dataset[i]["probable"], dataset[i]["base"], suffix=False))
            improbable_surprisal.append(process_prompt(model, dataset[i]["improbable"], dataset[i]["base"], suffix=False))
            impossible_surprisal.append(process_prompt(model, dataset[i]["impossible"], dataset[i]["base"], suffix=False))
            inconceivable_surprisal.append(process_prompt(model, dataset[i]["inconceivable"], dataset[i]["base"], suffix=False))


    return {
        "probable": probable_surprisal,
        "improbable": improbable_surprisal,
        "impossible": impossible_surprisal,
        "inconceivable": inconceivable_surprisal,
    }

def plot_surprisal(results, figfolder, model_name, suffix=True):

    n_per_condition = len(results["probable"])
    condition = ["probable"] * n_per_condition +\
    ["improbable"] * n_per_condition +\
    ["impossible"] * n_per_condition +\
    ["inconceivable"] * n_per_condition 

    surprisal = results["probable"] + results["improbable"] + results["impossible"] +\
        results["inconceivable"]
    
    df = pd.DataFrame.from_dict({"condition": condition, "surprisal": surprisal})
    sns.barplot(df, x="condition", y="surprisal")
    if suffix:
        plt.title(f"{model_name}: Suffix Surprisal by Condition")
        plt.savefig(os.path.join(figfolder, "surprisal_suffix.png"))
    else:
        plt.title(f"{model_name}: Surprisal by Condition")
        plt.savefig(os.path.join(figfolder, "surprisal.png"))    


def run_knn_analysis(results, figfolder):
    knn = TieBreakerKNN(n_neighbors=3)

    x = results["probable"] + results["improbable"] + results["impossible"] +\
    results["inconceivable"]
    x = np.array(x)

    n_per_condition = len(results["probable"])
    y = [0] * n_per_condition +\
        [1] * n_per_condition +\
        [2] * n_per_condition +\
        [3] * n_per_condition 
    y = np.array(y)
    
    # Save Preds for confusion_matrix
    y_gt = []
    y_pred = []

    # Leave-one-out CV
    cv = LeaveOneOut()
    for train_idxs, test_idx in cv.split(x):
        x_train, x_test = x[train_idxs], x[test_idx]
        y_train, y_test = y[train_idxs], y[test_idx]

        knn.fit(x_train.reshape(-1, 1), y_train)
        y_pred.append(knn.predict(x_test.reshape(-1, 1))[0])
        y_gt.append(y_test[0])

    confusion = confusion_matrix(y_gt, y_pred, normalize="true")

    labels = ["probable", "improbable", "impossible", "inconceivable"]

    # Plot results
    plt.figure()
    g = sns.heatmap(
        data=confusion,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        cmap=sns.color_palette("light:#5A9", as_cmap=True),
    )
    plt.ylabel("Correct Class")
    plt.xlabel("Predicted Class")
    plt.title(f"Surprisal Confusion Matrix")
    plt.savefig(os.path.join(figfolder, "surprisal_confusion.png"))
    plt.close()

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
    results = surprisal_eval(model, dataset, just_suffix=args.suffix)

    # Plot results
    plot_surprisal(results, figfolder, args.model, suffix=args.suffix)

    run_knn_analysis(results, figfolder)
