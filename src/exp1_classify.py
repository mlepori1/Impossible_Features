"""
This file is used to classify stimulus pairs using either vector projections
or probability estimates
"""

import os
import pickle as pkl

import pandas as pd
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

import utils


DATASET2COMPARISONS = {
    "goulding/data.csv": [
        ("probable", "improbable"),
        ("probable", "impossible"),
        ("improbable", "impossible"),
    ],
    "vega_mendoza/data.csv": [
        ("probable", "improbable_unrelated"),
        ("probable", "improbable_related"),
        ("probable", "inconceivable_unrelated"),
        ("probable", "inconceivable_related"),
        ("improbable_unrelated", "inconceivable_unrelated"),
        ("improbable_unrelated", "inconceivable_related"),
        ("improbable_related", "inconceivable_unrelated"),
        ("improbable_related", "inconceivable_related"),
    ],
    "kauf/DTFit_data.csv": [
        ("probable", "improbable"),
    ],
    "kauf/EventsAdapt_data.csv": [
        ("probable", "improbable"),
        ("probable", "inconceivable"),
    ],
    "kauf/EventsRev_data.csv": [
        ("probable", "improbable"),
    ],
}


def load_representation(model, classifier_type, comparison):
    # Load vector representation for making a particular comparison
    feature_0 = comparison[0].split("_")[0]  # In case it's from vega_mendoza
    feature_1 = comparison[1].split("_")[0]
    rep_prefix = feature_0 + "_" + feature_1
    all_rep_files = os.listdir(
        f"../artifacts/{model}/Linear_Representation/{classifier_type}/"
    )

    rep_file = list(filter(lambda x: rep_prefix in x, all_rep_files))[0]
    layer = int(rep_file.split("_")[-1][:-4])

    return (
        pkl.load(
            open(
                os.path.join(
                    f"../artifacts/{model}/Linear_Representation/{classifier_type}/{rep_file}"
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
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model"])

    ### Load Dataset
    data = pd.read_csv(
        os.path.join("..", "data", "classification", config["dataset_path"])
    )
    comparisons = DATASET2COMPARISONS[config["dataset_path"]]

    ### Process Dataset
    # For linear representation classifiers: feature -> layer -> item_set_id -> state
    # For probability classifiers: feature -> item_set_id -> probability
    features2processed = {}

    features = []
    for comparison in comparisons:
        for feature in comparison:
            features.append(feature)

    # Remove duplicates
    features = list(set(features))

    for feature in features:

        if "_" in feature:
            metadata = feature.split("_")[-1]
            base_feature = feature.split("_")[0]
            current_data = data[data["metadata"] == metadata]
        else:
            base_feature = feature
            current_data = data

        if config["classifier_type"] in ["Linear_Representation", "PC", "Random"]:
            features2processed[feature] = utils.compute_hidden_states(
                model, tokenizer, current_data, base_feature
            )
        elif config["classifier_type"] == "Probability":
            features2processed[feature] = (
                utils.compute_summed_log_probs_for_classification(
                    model, tokenizer, current_data, base_feature
                )
            )
        else:
            raise ValueError()

    ### Evaluate each comparison
    df = {"comparison": [], "accuracy": []}

    for comparison in tqdm(comparisons, desc="Processing Comparisons"):

        comparison_accuracy = []

        ### Vector-based classification
        if config["classifier_type"] in ["Linear_Representation", "PC", "Random"]:
            representation, layer = load_representation(
                config["model"], config["classifier_type"], comparison
            )

            modal_feature_0 = features2processed[comparison[0]]
            modal_feature_1 = features2processed[comparison[1]]

            # Find item_sets that contain both modal features to ensure minimal pairs
            item_sets = list(
                set(modal_feature_0[layer].keys()).intersection(
                    set(modal_feature_1[layer].keys())
                )
            )

            for item_set_idx in item_sets:
                correct_bool, _ = utils.classify_vector(
                    modal_feature_1[layer][item_set_idx],  # Should be higher
                    modal_feature_0[layer][item_set_idx],  # Should be lower
                    representation,  # Vector for classifying
                    unit_norm=False,
                )
                comparison_accuracy.append(correct_bool)

        ### Log-Prob based classification
        if config["classifier_type"] == "Probability":
            modal_feature_0 = features2processed[comparison[0]]
            modal_feature_1 = features2processed[comparison[1]]

            # Find item_sets that contain both modal features
            item_sets = list(
                set(modal_feature_0.keys()).intersection(set(modal_feature_1.keys()))
            )

            for item_set_idx in item_sets:
                correct_bool = (
                    modal_feature_0[item_set_idx] > modal_feature_1[item_set_idx]
                )
                comparison_accuracy.append(correct_bool)

        df["comparison"].append("_".join(comparison))
        df["accuracy"].append(
            np.sum(comparison_accuracy).item() / len(comparison_accuracy)
        )

    pd.DataFrame.from_dict(df).to_csv(
        os.path.join(config["results_path"], config["classifier_type"] + ".csv"),
        index=False,
    )
