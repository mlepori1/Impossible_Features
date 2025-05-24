"""
This file is used to generate the difference vectors
that support the remaining analyses in the study.
"""

import argparse
import os
from collections import defaultdict
import pickle as pkl

import pandas as pd
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

import utils


DIFFERENCE_FEATURES = [
    ("probable", "improbable"),
    ("probable", "impossible"),
    ("probable", "inconceivable"),
    ("improbable", "impossible"),
    ("improbable", "inconceivable"),
    ("impossible", "inconceivable"),
]


def generate_mean_diff_vector(train_splits, representation_type):
    """Compute the difference vector of hidden states of sentences exhibiting two features"""

    layer2diffs = defaultdict(list)
    for split in train_splits:
        modal_feature_0 = split[representation_type[0]]
        modal_feature_1 = split[representation_type[1]]

        for layer in modal_feature_1.keys():
            for item_set_idx in modal_feature_1[layer].keys():
                layer2diffs[layer].append(
                    (
                        modal_feature_1[layer][item_set_idx]
                        - modal_feature_0[layer][item_set_idx]
                    ).reshape(-1)
                )

    layer2meandiff = {}

    for layer in layer2diffs.keys():
        mean_diff = np.mean(np.stack(layer2diffs[layer], axis=0), axis=0)
        layer2meandiff[layer] = mean_diff

    return layer2meandiff


def get_median_layer(layers):
    # Helper function to return the median of the
    # list of best layers, but make sure
    # that the returned value is in the list
    layers = np.sort(layers)
    middle_idx = len(layers) // 2
    if len(layers) % 2 == 0:
        return layers[middle_idx - 1]
    else:
        return layers[middle_idx]


def linear_representation_cv(splits2features2states, linear_representation_type):
    ### Run CV for linear representations of features
    layer2results = defaultdict(list)

    for test_idx in range(5):
        test_split = splits2features2states[test_idx]
        train_split_idxs = list(range(5))
        train_split_idxs.remove(test_idx)
        train_split = [splits2features2states[idx] for idx in train_split_idxs]

        layer2linear = generate_mean_diff_vector(
            train_split, linear_representation_type
        )

        split_results = test_linear_representation(
            test_split, layer2linear, linear_representation_type
        )
        for layer in split_results.keys():
            layer2results[layer].append(split_results[layer])

    ### Identify layer with best diff_vector for this concept,
    #   return that vector, layer, and results

    representation_accuracies = {
        "layer": [],
        "accuracy": [],
    }

    for layer in range(len(layer2results.keys())):
        accuracy = np.mean(layer2results[layer])
        representation_accuracies["layer"].append(layer)
        representation_accuracies["accuracy"].append(accuracy)

    representation_accuracies = pd.DataFrame.from_dict(representation_accuracies)

    # If multiple best rows, take the median row
    best_layer = get_median_layer(
        representation_accuracies[
            representation_accuracies["accuracy"]
            == representation_accuracies["accuracy"].max()
        ]["layer"]
    )

    # Generate diff vectors using complete dataset
    final_layer2linear = generate_mean_diff_vector(
        splits2features2states, linear_representation_type
    )
    representation = final_layer2linear[best_layer]

    return representation, best_layer, layer2results


def test_linear_representation(test_split, layer2linear, representation_type):
    layer2result = {}

    modal_feature_0 = test_split[representation_type[0]]
    modal_feature_1 = test_split[representation_type[1]]

    for layer in layer2linear.keys():
        layer_results = []
        for item_set_idx in modal_feature_1[layer].keys():
            correct_bool, _ = utils.classify_vector(
                modal_feature_1[layer][item_set_idx],  # Expect higher
                modal_feature_0[layer][item_set_idx],  # Expect Lower
                layer2linear[layer],  # Vector for classifying,
                unit_norm=False,
            )
            layer_results.append(correct_bool)

        # Report average accuracy between sentence projections
        layer2result[layer] = {}
        layer2result[layer] = np.sum(layer_results).item() / len(layer_results)

    return layer2result


def pc_cv(splits2features2states, model_name, representation_type):
    ### Run CV for principle components of hidden states
    layer2pc2results = defaultdict(lambda: defaultdict(list))

    for test_idx in range(5):
        test_split = splits2features2states[test_idx]
        split_results = test_pc(test_split, model_name, representation_type)
        for layer in split_results.keys():
            for pc_num in range(0, 3):
                layer2pc2results[layer][pc_num].append(split_results[layer][pc_num])

    ### Identify layer and PC with best accuracy for this concept,
    #   return that PCA object, layer, PC number, and results
    pc_accuracies = {
        "layer": [],
        "pc": [],
        "accuracy": [],
        "flip": [],
    }

    flip_dict = {}
    # Start at 1, because layer 0 trivially gets 0% accuracy -- each token is identical
    for layer in range(1, len(layer2pc2results.keys())):
        flip_dict[layer] = {}
        for pc_num in range(0, 3):
            accuracy = np.mean(layer2pc2results[layer][pc_num])
            # If accuracy is < .5, flip the PC
            flip = False
            if accuracy < 0.5:
                accuracy = 1.0 - accuracy
                flip = True
            flip_dict[layer][pc_num] = flip

            pc_accuracies["layer"].append(layer)
            pc_accuracies["pc"].append(pc_num)
            pc_accuracies["accuracy"].append(accuracy)
            pc_accuracies["flip"].append(flip)

    pc_accuracies = pd.DataFrame.from_dict(pc_accuracies)

    # If multiple best rows, choose median layer
    best_rows = pc_accuracies[
        pc_accuracies["accuracy"] == pc_accuracies["accuracy"].max()
    ]
    best_layer = get_median_layer(best_rows["layer"])
    best_row = best_rows[best_rows["layer"] == best_layer].iloc[
        0
    ]  # .iloc[0] In case multiple PCs get same accuracy

    best_pc = best_row["pc"]
    flip_pc = best_row["flip"]

    # Load up relevant PCA, return first 3 PCs
    pca = pkl.load(
        open(f"../artifacts/{model_name}/PCA/Layer_{best_layer}_PCA.pkl", "rb")
    )
    pc = pca.components_[best_pc]
    if flip_pc:
        pc = -1 * pc

    return pc, best_layer, best_pc, layer2pc2results, flip_dict


def test_pc(test_split, model_name, representation_type):
    layer2result = {}

    modal_feature_0 = test_split[representation_type[0]]
    modal_feature_1 = test_split[representation_type[1]]

    for layer in modal_feature_0.keys():
        layer2result[layer] = {}

        pca = pkl.load(
            open(f"../artifacts/{model_name}/PCA/Layer_{layer}_PCA.pkl", "rb")
        )
        # For each layer, report the accuracy of top 3 PCs
        for pc_number in range(0, 3):
            pc_accs = []
            pc = pca.components_[pc_number]
            for item_set_idx in modal_feature_1[layer].keys():
                # Uncertain which should be higher, heuristically assume this one. Report whichever direction works best,
                # and flip the PC accordingly
                correct_bool, _ = utils.classify_vector(
                    modal_feature_1[layer][item_set_idx],
                    modal_feature_0[layer][item_set_idx],
                    pc,  # Vector for classifying
                    unit_norm=False,
                )
                pc_accs.append(correct_bool)
            layer2result[layer][pc_number] = np.sum(pc_accs) / len(pc_accs)

    return layer2result


def random_cv(splits2features2states, representation_type, num_layers, hidden_size):
    ### Run CV for random directions
    layer2results = defaultdict(list)
    random_vectors = np.random.rand(num_layers + 1, hidden_size)

    for test_idx in range(5):
        test_split = splits2features2states[test_idx]
        split_results = test_random(test_split, random_vectors, representation_type)
        for layer in split_results.keys():
            layer2results[layer].append(split_results[layer])

    ### Identify layer with best accuracy for this concept,
    #   return that vector, layer, and results
    random_accuracies = {
        "layer": [],
        "accuracy": [],
        "flip": [],
    }

    flip_dict = {}
    # Start at 1, because layer 0 trivially gets 0% accuracy -- each token is identical
    for layer in range(1, len(layer2results.keys())):
        flip_dict[layer] = {}
        accuracy = np.mean(layer2results[layer])

        # If accuracy is < .5, flip the PC
        flip = False
        if accuracy < 0.5:
            accuracy = 1.0 - accuracy
            flip = True
        flip_dict[layer] = flip

        random_accuracies["layer"].append(layer)
        random_accuracies["accuracy"].append(accuracy)
        random_accuracies["flip"].append(flip)

    random_accuracies = pd.DataFrame.from_dict(random_accuracies)

    # If multiple best rows, choose median layer
    best_rows = random_accuracies[
        random_accuracies["accuracy"] == random_accuracies["accuracy"].max()
    ]
    best_layer = get_median_layer(best_rows["layer"])
    best_row = best_rows[best_rows["layer"] == best_layer].iloc[0]

    flip_pc = best_row["flip"]

    vector = random_vectors[best_layer]
    if flip_pc:
        vector = -1 * random_vectors[best_layer]

    return vector, best_layer, layer2results, flip_dict


def test_random(test_split, random_vectors, representation_type):
    layer2result = {}

    modal_feature_0 = test_split[representation_type[0]]
    modal_feature_1 = test_split[representation_type[1]]

    for layer in modal_feature_0.keys():
        layer2result[layer] = []
        vector = random_vectors[layer]
        for item_set_idx in modal_feature_1[layer].keys():
            # Uncertain which should be higher, heuristically assume this one. Report whichever direction works best,
            # and flip the vector accordingly
            correct_bool, _ = utils.classify_vector(
                modal_feature_1[layer][item_set_idx],
                modal_feature_0[layer][item_set_idx],
                vector,  # Vector for classifying
                unit_norm=False,
            )
            layer2result[layer].append(correct_bool)
        layer2result[layer] = np.sum(layer2result[layer]) / len(layer2result[layer])

    return layer2result


def probability_cv(splits2features2probs, representation_type):
    ### Compute Probability-based classification accuracy for all splits
    results = []
    for test_idx in range(5):
        split_results = []
        test_split = splits2features2probs[test_idx]
        modal_feature_0 = test_split[representation_type[0]]
        modal_feature_1 = test_split[representation_type[1]]

        for item_set_idx in modal_feature_1.keys():
            # Expect modal_feature_0 probabilities to be higher than modal_feature_1
            correct_bool = modal_feature_0[item_set_idx] > modal_feature_1[item_set_idx]
            split_results.append(correct_bool)
        results.append(np.mean(split_results).item())
    return results


if __name__ == "__main__":
    config = utils.get_config()

    ### Set Up Output
    os.makedirs(
        os.path.join(config["artifact_path"], "Linear_Representation"), exist_ok=True
    )
    os.makedirs(os.path.join(config["artifact_path"], "PC"), exist_ok=True)
    os.makedirs(os.path.join(config["artifact_path"], "Random"), exist_ok=True)

    os.makedirs(
        os.path.join(config["results_path"], "Linear_Representation"), exist_ok=True
    )
    os.makedirs(os.path.join(config["results_path"], "PC"), exist_ok=True)
    os.makedirs(os.path.join(config["results_path"], "Random"), exist_ok=True)
    os.makedirs(os.path.join(config["results_path"], "Probability"), exist_ok=True)

    ### Set up model
    torch.set_grad_enabled(False)
    torch.manual_seed(19)
    np.random.seed(19)

    # For analyses over training time
    if "revision" in config.keys():
        revision = config["revision"]
    else:
        revision = "main"

    model = AutoModelForCausalLM.from_pretrained(
        config["model"],
        torch_dtype=utils.strtype2torchtype(config["dtype"]),
        device_map="auto",
        revision=revision,
        token="TOKEN",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"],
        revision=revision,
        token="TOKEN",
    )

    ### Load Dataset and split
    data = pd.read_csv("../data/classification/hu_shades/data.csv")
    item_set_ids = data["item_set_id"].unique()
    np.random.shuffle(item_set_ids)
    split_ids = np.array_split(item_set_ids, 5)

    splits = []
    for split_id_set in split_ids:
        splits.append(data[data["item_set_id"].isin(split_id_set)])

    ### Compute and cache hidden states, log_probs
    splits2features2states = []
    splits2features2probs = []

    for split_number, split in enumerate(tqdm(splits)):

        splits2features2states.append({})
        splits2features2probs.append({})

        split = split.sort_values(by="item_set_id", ignore_index=True)
        # Compute hidden states
        splits2features2states[split_number]["probable"] = utils.compute_hidden_states(
            model, tokenizer, split, "probable"
        )
        splits2features2states[split_number]["improbable"] = (
            utils.compute_hidden_states(model, tokenizer, split, "improbable")
        )
        splits2features2states[split_number]["impossible"] = (
            utils.compute_hidden_states(model, tokenizer, split, "impossible")
        )
        splits2features2states[split_number]["inconceivable"] = (
            utils.compute_hidden_states(model, tokenizer, split, "inconceivable")
        )
        # Compute log-probs
        splits2features2probs[split_number]["probable"] = (
            utils.compute_summed_log_probs_for_classification(
                model, tokenizer, split, "probable"
            )
        )
        splits2features2probs[split_number]["improbable"] = (
            utils.compute_summed_log_probs_for_classification(
                model, tokenizer, split, "improbable"
            )
        )
        splits2features2probs[split_number]["impossible"] = (
            utils.compute_summed_log_probs_for_classification(
                model, tokenizer, split, "impossible"
            )
        )
        splits2features2probs[split_number]["inconceivable"] = (
            utils.compute_summed_log_probs_for_classification(
                model, tokenizer, split, "inconceivable"
            )
        )

    ### Perform 5-Fold CV on modal linear representations, finding the best layer for each vector
    for representation_type in DIFFERENCE_FEATURES:
        representation, rep_layer, results = linear_representation_cv(
            splits2features2states, representation_type
        )
        # Save results and artifacts
        df = {
            "layer": [],
            "split": [],
            "accuracy": [],
        }
        for layer in results.keys():
            layer_results = results[layer]
            for split_idx in range(len(layer_results)):
                df["layer"].append(layer)
                df["split"].append(split_idx)
                df["accuracy"].append(layer_results[split_idx])

        pd.DataFrame.from_dict(df).to_csv(
            os.path.join(
                config["results_path"],
                "Linear_Representation",
                representation_type[0] + "_" + representation_type[1] + ".csv",
            ),
            index=False,
        )

        artifact_path = os.path.join(
            config["artifact_path"],
            "Linear_Representation",
            representation_type[0]
            + "_"
            + representation_type[1]
            + f"_Layer_{rep_layer}.pkl",
        )
        pkl.dump(representation, open(artifact_path, "wb"))

    if not "only_linear" in config.keys() or not config["only_linear"]:
        ### Perform 5-Fold CV for projections on the first 3 PCs of each layer, finding the best layer and PC for each concept
        for representation_type in DIFFERENCE_FEATURES:
            pc, best_pc_layer, best_pc_num, results, flip_dict = pc_cv(
                splits2features2states, config["model"], representation_type
            )
            # Save results and artifacts
            df = {
                "layer": [],
                "pc": [],
                "split": [],
                "accuracy": [],
            }
            # Don't record the 0th layer, as it is identical across stimuli
            layers = list(results.keys())
            layers.remove(0)

            for layer in layers:
                layer_results = results[layer]
                for pc_num in range(3):
                    pc_results = layer_results[pc_num]
                    for split_idx in range(len(pc_results)):
                        df["layer"].append(layer)
                        df["pc"].append(pc_num)
                        df["split"].append(split_idx)

                        if flip_dict[layer][pc_num]:
                            df["accuracy"].append(1 - pc_results[split_idx])
                        else:
                            df["accuracy"].append(pc_results[split_idx])

            pd.DataFrame.from_dict(df).to_csv(
                os.path.join(
                    config["results_path"],
                    "PC",
                    representation_type[0] + "_" + representation_type[1] + ".csv",
                ),
                index=False,
            )

            artifact_path = os.path.join(
                config["artifact_path"],
                "PC",
                representation_type[0]
                + "_"
                + representation_type[1]
                + f"_PC_{best_pc_num}_Layer_{best_pc_layer}.pkl",
            )
            pkl.dump(pc, open(artifact_path, "wb"))

        # As a control, perform 5-fold CV using random vectors
        # Get model metadata from different types of configs
        if hasattr(model.config, "text_config"):
            num_layers = model.config.text_config.num_hidden_layers
            hidden_size = model.config.text_config.hidden_size
        else:
            num_layers = model.config.num_hidden_layers
            hidden_size = model.config.hidden_size

        for representation_type in DIFFERENCE_FEATURES:
            vector, best_layer, results, flip_dict = random_cv(
                splits2features2states,
                representation_type,
                num_layers,
                hidden_size,
            )
            # Save results and artifacts
            df = {
                "layer": [],
                "split": [],
                "accuracy": [],
            }

            # Don't record the 0th layer, as it is identical across stimuli
            layers = list(results.keys())
            layers.remove(0)

            for layer in layers:
                layer_results = results[layer]
                for split_idx in range(len(layer_results)):
                    df["layer"].append(layer)
                    df["split"].append(split_idx)

                    if flip_dict[layer]:
                        df["accuracy"].append(1 - layer_results[split_idx])
                    else:
                        df["accuracy"].append(layer_results[split_idx])

            pd.DataFrame.from_dict(df).to_csv(
                os.path.join(
                    config["results_path"],
                    "Random",
                    representation_type[0] + "_" + representation_type[1] + ".csv",
                ),
                index=False,
            )

            artifact_path = os.path.join(
                config["artifact_path"],
                "Random",
                representation_type[0]
                + "_"
                + representation_type[1]
                + f"_Layer_{best_layer}.pkl",
            )
            pkl.dump(vector, open(artifact_path, "wb"))

        ### Get comparable 5-Fold CV accuracy for probability-based predictions
        for representation_type in DIFFERENCE_FEATURES:
            results = probability_cv(splits2features2probs, representation_type)
            # Save results
            df = {
                "split": [],
                "accuracy": [],
            }

            for split_idx in range(len(results)):
                df["split"].append(split_idx)
                df["accuracy"].append(results[split_idx])

            pd.DataFrame.from_dict(df).to_csv(
                os.path.join(
                    config["results_path"],
                    "Probability",
                    representation_type[0] + "_" + representation_type[1] + ".csv",
                ),
                index=False,
            )
