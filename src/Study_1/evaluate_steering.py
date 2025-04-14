"""This file uses GPT-4 to evaluate the results
of steering a model
"""

from openai import OpenAI
from tqdm import tqdm

import pandas as pd
import numpy as np

import argparse
import os
from collections import defaultdict


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
        "--val_size",
        default=10,
        type=int,
        help="Number of prefixes for tuning multiplier",
    )

    parser.add_argument(
        "--test_size",
        default=30,
        type=int,
        help="Number of prefixes for tuning multiplier",
    )

    parser.add_argument(
        "--top_N",
        default=20,
        type=int,
        help="How much of the distribution to look at when evalutating",
    )

    args = parser.parse_args()
    return args


def classify(examples, generation, concept, description):
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f'Help me classify some scenarios according to whether or not they are \
                {concept}.\n{description}\n\
                Here are some examples:\n{examples}\n\
                Now it\'s your turn! Note that the description of this particular scenario may not be complete. Just answer as best as you can. Please only respond with either "yes" or "no"!\n\
                SCENARIO: {generation} CLASS:',
            },
        ],
    )
    response = completion.choices[0].message.content.lower()
    return response


def get_classification_examples(N=10):
    """Generate classification examples using
    shades data to calibrate GPT-4 autorater
    """

    shades_dataset = pd.read_csv("../data/shades_of_zero.csv")
    shades_dataset = shades_dataset.sample(frac=1, random_state=19).reset_index(drop=True)
    classes = (
        ["probable"] * N
        + ["improbable"] * N
        + ["impossible"] * N
        + ["inconceivable"] * N
    )
    np.random.seed(19)
    np.random.shuffle(classes)

    examples_dict = {
        "improbable": "",
        "impossible": "",
        "inconceivable": "",
    }

    for idx, row in shades_dataset.iterrows():
        if idx == len(classes):
            break

        prefix = row["classification_prefix"].replace("[POSS]", "their").strip()
        continuation = row[classes[idx]].replace("[POSS]", "their").strip()
        example = (
            "SCENARIO: "
            + prefix
            + " "
            + continuation
            + ". CLASS: "
        )

        if classes[idx] == "improbable":
            examples_dict["improbable"] += example + "yes\n"
            examples_dict["impossible"] += example + "no\n"
            examples_dict["inconceivable"] += example + "no\n"
        if classes[idx] == "impossible":
            examples_dict["improbable"] += example + "no\n"
            examples_dict["impossible"] += example + "yes\n"
            examples_dict["inconceivable"] += example + "no\n"
        if classes[idx] == "inconceivable":
            examples_dict["improbable"] += example + "no\n"
            examples_dict["impossible"] += example + "no\n"
            examples_dict["inconceivable"] += example + "yes\n"
        if classes[idx] == "probable":
            examples_dict["improbable"] += example + "no\n"
            examples_dict["impossible"] += example + "no\n"
            examples_dict["inconceivable"] += example + "no\n"

    return examples_dict


def tune_multiplier(data, examples_dict, descriptions):
    """Have GPT-4 Classify the generations of a subset
    of prefixes using all multipliers. For each steering concept,
    find the multiplier that results in the largest average proportion of
    generations that exhibit the desired class
    """
    prefixes = data["prefix"].unique()

    results = {
        "improbable": defaultdict(list),
        "impossible": defaultdict(list),
        "inconceivable": defaultdict(list),
    }

    multipliers = None

    # Classify generations
    for steering in ["improbable", "impossible", "inconceivable"]:
        steering_subset = data[data["steering_concept"] == steering]
        multipliers = steering_subset["multiplier"].unique()
        for multiplier in multipliers:
            multiplier_subset = steering_subset[
                steering_subset["multiplier"] == multiplier
            ]
            for prefix in tqdm(
                prefixes, desc=f"Processing {steering}, Multiplier {multiplier}"
            ):
                prefix_subset = multiplier_subset[multiplier_subset["prefix"] == prefix]
                success = []
                for _, row in prefix_subset.iterrows():
                    generation = row["prefix"] + row["generation"]
                    response = classify(examples_dict[steering], generation, steering, descriptions[steering])

                    if "yes" in response:
                        success.append(1)
                    else:
                        success.append(0)
                results[steering][multiplier].append(np.mean(success).item())
            results[steering][multiplier] = np.mean(
                results[steering][multiplier]
            ).item()

    # Pick out the multiplier that works best for each steering vector
    multiplier_results = {}
    for steering in ["improbable", "impossible", "inconceivable"]:
        best_multiplier = None
        most_success = -1
        for multiplier in multipliers:
            if results[steering][multiplier] > most_success:
                most_success = results[steering][multiplier]
                best_multiplier = multiplier
        multiplier_results[steering] = (best_multiplier, most_success)

    return multiplier_results


def run_evaluation(data, examples_dict, descriptions, multipliers):
    """Have GPT-4 Classify the generations of a subset
    of prefixes using the best multiplier per class
    """
    prefixes = data["prefix"].unique()

    ### First, compute the baseline rate of each category for each prefix
    baseline_results = {
        "improbable": [],
        "impossible": [],
        "inconceivable": [],
    }

    baseline_subset = data[data["steering_concept"].isna()]
    for steering in baseline_results.keys():
        for prefix in tqdm(prefixes, "Running Baseline"):
            prefix_subset = baseline_subset[baseline_subset["prefix"] == prefix]
            success = []

            for _, row in prefix_subset.iterrows():
                generation = row["prefix"] + row["generation"]
                response = classify(examples_dict, generation, steering, descriptions[steering])

                if "yes" in response:
                    success.append(1)
                else:
                    success.append(0)

            baseline_results[steering].append(np.mean(success).item())

    baseline_results = {k: np.mean(v).item() for k, v in baseline_results.items()}

    ### Next, compute the rates of each category after steering
    results = {
        "improbable": [],
        "impossible": [],
        "inconceivable": [],
    }

    # Classify generations
    for steering in ["improbable", "impossible", "inconceivable"]:
        steering_subset = data[data["steering_concept"] == steering]
        multiplier_subset = steering_subset[
            steering_subset["multiplier"] == multipliers[steering][0]
        ]
        for prefix in tqdm(prefixes, desc=f"Running {steering} evaluation"):
            prefix_subset = multiplier_subset[multiplier_subset["prefix"] == prefix]
            success = []
            for _, row in prefix_subset.iterrows():
                generation = row["prefix"] + row["generation"]
                response = classify(examples_dict[steering], generation, steering, descriptions[steering])

                if "yes" in response:
                    success.append(1)
                else:
                    success.append(0)
                results[steering].append(np.mean(success).item())
    results = {k: np.mean(v).item() for k, v in results.items()}

    return results, baseline_results


def process_results(multipliers, results, baseline_results, top_n, outfolder):
    """Package results and save dataframe
    """
    df = {
        "steering": [],
        "concept": [],
        "success_rate": [],
        "top_N": [],
        "multiplier": [],
    }

    for concept, success_rate in baseline_results.items():
        df["steering"].append(False)
        df["concept"].append(concept)
        df["success_rate"].append(success_rate)
        df["top_N"].append(top_n)
        df["multiplier"].append(multipliers[concept][0])

    for concept, success_rate in results.items():
        df["steering"].append(True)
        df["concept"].append(concept)
        df["success_rate"].append(success_rate)
        df["top_N"].append(top_n)
        df["multiplier"].append(multipliers[concept][0])
    pd.DataFrame.from_dict(df).to_csv(
        os.path.join(outfolder, "steering_evaluation.csv"), index=False
    )


if __name__ == "__main__":

    args = parse_arguments()
    outfolder = os.path.join(args.outfolder, args.model)

    steering_dataset = pd.read_csv(os.path.join(outfolder, "steering_generations.csv"))
    steering_dataset = steering_dataset[steering_dataset["rank"] < args.top_N]

    examples_dict = get_classification_examples()
    descriptions = {
        "improbable": "Improbable means that the scenario is possible in the real world, but that it is not very likely.",
        "impossible": "Impossible means that the scenario cannot happen in the real world, but that it might happen in some imaginary world.",
        "inconceivable": "Inconceivable means that the scenario cannot happen due to some conceptual error.",
    }

    prefixes = steering_dataset["prefix"].unique()

    val_prefixes = prefixes[: args.val_size]
    test_prefixes = prefixes[args.val_size :args.val_size + args.test_size]

    val_data = steering_dataset[steering_dataset["prefix"].isin(val_prefixes)]
    test_data = steering_dataset[steering_dataset["prefix"].isin(test_prefixes)]

    #multipliers = tune_multiplier(val_data, examples_dict, descriptions)
    multipliers = {
        "improbable": (5, None),
        "impossible": (5, None),
        "inconceivable": (5, None),
    }
    print(multipliers)
    results, baseline_results = run_evaluation(test_data, examples_dict, descriptions, multipliers)

    process_results(multipliers, results, baseline_results, args.top_N, outfolder)
