"""
This file is used to correlate projections along difference vectors with
each other, and with interpretable features.

It correlates vector projections with features from Hu et al. 2024 and 2025, as
well as Tuckute et al 2024
"""

import os
import pickle as pkl

import pandas as pd
import numpy as np
import scipy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import utils


VECTORS = ["probable_improbable", "improbable_impossible", "impossible_inconceivable"]


def load_representation(model, rep_prefix):
    # Load vector representation for making a particular comparison
    all_rep_files = os.listdir(
        f"../artifacts/{model}/Linear_Representation/Linear_Representation/"
    )

    rep_file = list(filter(lambda x: rep_prefix in x, all_rep_files))[0]
    layer = int(rep_file.split("_")[-1][:-4])

    return (
        pkl.load(
            open(
                os.path.join(
                    f"../artifacts/{model}/Linear_Representation/Linear_Representation/{rep_file}"
                ),
                "rb",
            )
        ),
        layer,
    )


if __name__ == "__main__":
    config = utils.get_config()

    ### Set Up Output
    os.makedirs(os.path.join(config["results_path"]), exist_ok=True)

    ### Set up model
    torch.set_grad_enabled(False)
    torch.manual_seed(19)
    np.random.seed(19)

    model = AutoModelForCausalLM.from_pretrained(
        config["model"],
        torch_dtype=utils.strtype2torchtype(config["dtype"]),
        device_map="auto",
        token="TOKEN",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"],
        token="TOKEN",
    )

    ### Load representations for projections
    vectors = [load_representation(config["model"], vector) for vector in VECTORS]

    ### Vector to Vector Correlations
    ### Process Tuckute Dataset
    data = pd.read_csv(os.path.join("..", "data", "correlation", "tuckute", "data.csv"))

    df = {
        "Vector 1": [],
        "Vector 2": [],
        "Pearson r": [],
    }

    projections = utils.project_data(data, model, tokenizer, vectors)

    # Compute Vector to Vector Correlations
    r_0_1 = scipy.stats.pearsonr(projections[0], projections[1]).statistic
    df["Vector 1"].append("Probable-Improbable")
    df["Vector 2"].append("Improbable-Impossible")
    df["Pearson r"].append(r_0_1)

    r_0_2 = scipy.stats.pearsonr(projections[0], projections[2]).statistic
    df["Vector 1"].append("Probable-Improbable")
    df["Vector 2"].append("Impossible-Inconceivable")
    df["Pearson r"].append(r_0_2)

    r_1_2 = scipy.stats.pearsonr(projections[1], projections[2]).statistic
    df["Vector 1"].append("Improbable-Impossible")
    df["Vector 2"].append("Impossible-Inconceivable")
    df["Pearson r"].append(r_1_2)

    pd.DataFrame.from_dict(df).to_csv(
        os.path.join(config["results_path"], "Pairwise_Correlations.csv"), index=False
    )

    ### Vector to feature Correlations
    ### Start with the Tuckute dataset because it is processed already
    df = {
        "Vector": [],
        "Feature": [],
        "Pearson r": [],
    }

    features = data.columns.tolist()
    features.remove("sentence")

    for vect_idx, vector in enumerate(VECTORS):
        for feature in features:
            df["Vector"].append(vector)
            df["Feature"].append(feature)
            df["Pearson r"].append(
                scipy.stats.pearsonr(projections[vect_idx], data[feature]).statistic
            )

    ### Process Shades Dataset
    data = pd.read_csv(
        os.path.join("..", "data", "correlation", "hu_shades", "data.csv")
    )
    projections = utils.project_data(data, model, tokenizer, vectors)

    for vect_idx, vector in enumerate(VECTORS):
        df["Vector"].append(vector)
        df["Feature"].append("Subjective Event Likelihood")
        df["Pearson r"].append(
            scipy.stats.pearsonr(
                projections[vect_idx], data["Subjective Event Likelihood"]
            ).statistic
        )

    ### Process Nonsense Dataset
    data = pd.read_csv(
        os.path.join("..", "data", "correlation", "hu_nonsense", "data.csv")
    )
    projections = utils.project_data(data, model, tokenizer, vectors)

    for vect_idx, vector in enumerate(VECTORS):
        df["Vector"].append(vector)
        df["Feature"].append("Ranked Inconceivability")
        df["Pearson r"].append(
            scipy.stats.pearsonr(
                projections[vect_idx], data["Ranked Inconceivability"]
            ).statistic
        )

    pd.DataFrame.from_dict(df).to_csv(
        os.path.join(config["results_path"], "Feature_Correlations.csv"), index=False
    )
