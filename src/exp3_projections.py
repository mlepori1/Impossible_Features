"""
This file is used project along difference vectors (or compute probabilities).
Then, see an analysis script to fit a LR model to these projections.
"""

import os
import pickle as pkl

import pandas as pd
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import utils


VECTORS = [
    "probable_improbable",
    "improbable_impossible",
    "impossible_inconceivable",
    "probable_impossible",
    "probable_inconceivable",
]


def load_representation(model, rep_prefix, representation_type):
    # Load vector representation for making a particular comparison
    all_rep_files = os.listdir(
        f"../artifacts/{model}/Linear_Representation/{representation_type}/"
    )

    rep_file = list(filter(lambda x: rep_prefix in x, all_rep_files))[0]
    layer = int(rep_file.split("_")[-1][:-4])

    return (
        pkl.load(
            open(
                os.path.join(
                    f"../artifacts/{model}/Linear_Representation/{representation_type}/{rep_file}"
                ),
                "rb",
            )
        ),
        layer,
    )


if __name__ == "__main__":
    config = utils.get_config()

    ### Set Up Output
    config["results_path"] = os.path.join(
        config["results_path"], config["dataset_path"][:-4]
    )
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

    ### Load in data
    data = pd.read_csv(
        os.path.join("..", "data", "calibration", config["dataset_path"])
    )

    if config["representation_type"] in ["Linear_Representation", "PC", "Random"]:
        ### Load representations for projections, and project
        vectors = [
            load_representation(config["model"], vector, config["representation_type"])
            for vector in VECTORS
        ]
        projections = utils.project_data(data, model, tokenizer, vectors)
        features = np.stack(projections, axis=-1)
        for vect_idx, vect in enumerate(VECTORS):
            data[vect] = features[:, vect_idx]

    elif config["representation_type"] in ["Probability"]:
        ### Compute sentence probabilities to use as a feature
        sentence_probabilities = utils.compute_summed_log_probs(data, model, tokenizer)
        features = np.array(sentence_probabilities).reshape(-1, 1)
        data["Probability"] = features[:, 0]

    else:
        raise ValueError()

    data.to_csv(
        os.path.join(config["results_path"], config["representation_type"] + ".csv"),
        index=False,
    )
