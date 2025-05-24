"""
This file is used project along difference vectors (or compute probabilities).
Then, fit a simple generative model to these projections, fit to experimenter-
defined labels. Use this model to generate probability distributions over labels
and compare to human-subjects classification data.
"""

import os
import pickle as pkl

import pandas as pd
import numpy as np

import scipy
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import mean_squared_error

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import utils


VECTORS = ["probable_improbable", "improbable_impossible", "impossible_inconceivable", "probable_impossible", "probable_inconceivable"]


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
    data = pd.read_csv(os.path.join("..", "data", "calibration", config["dataset_path"]))

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

    # ### Fit QDA Model
    # labels = data["label"]

    # qda = QDA().fit(features, labels)
    # qda_class_labels = list(qda.classes_)

    # ### Compute Projections on to compare the human-subjects classification behavior to the predictions from the
    # # generative model
    # kl_divs = []
    # mses = []
    # label_class_probs = []

    # for idx, row in data.iterrows():
    #     sample_features = features[idx]
    #     predicted_probabilities = qda.predict_proba(sample_features.reshape(1, -1))[0]
    #     true_probabilites = np.array(
    #         [row[class_label] for class_label in qda_class_labels]
    #     )

    #     kl_div = np.sum(scipy.special.rel_entr(true_probabilites, predicted_probabilities))
    #     mse = mean_squared_error(true_probabilites, predicted_probabilities)

    #     construction_label_idx = qda_class_labels.index(row["label"])
    #     prob_label_class = predicted_probabilities[construction_label_idx]

    #     kl_divs.append(kl_div)
    #     mses.append(mse)
    #     label_class_probs.append(prob_label_class)

    # ### Save off data
    # data["KL Div"] = kl_divs
    # data["MSE"] = mses
    # data["Label Class Probs"] = label_class_probs

    data.to_csv(os.path.join(config["results_path"], config["representation_type"] + ".csv"), index=False)
