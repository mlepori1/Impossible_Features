import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import json
import numpy as np
from collections import defaultdict
import copy

import matplotlib.pyplot as plt
import seaborn as sns



def exclude_attention(attention_checks):
    # assess attention checks
    attn_results = []

    for idx, attention_check in enumerate(attention_checks):
        response = attention_check["choices"][attention_check["response"]]
        attn_results.append(response == "Click This Button!")

    if not np.all(attn_results):
        return True
    else:
        return False
    
def process_critical_trials(critical_trials, results):
    pair2choices = {}
    for idx, trial in enumerate(critical_trials):
        response = trial["choices"][trial["response"]]
        pair_idx = trial["pair_idx"]
        condition = trial["condition_0"] + "_" + trial["condition_1"]
        concept = trial["concept"]
        results[concept][condition][pair_idx].append(response)
        pair2choices[pair_idx] = "_".join(trial["choices"])

    return results, pair2choices

def process_results(results, pair2stim):

    concept2bucket2acc = {}
    concept2pair2acc = {}
    concept2numraters = {}

    for concept, concept_dict in results.items():

        concept2bucket2acc[concept] = {}
        concept2pair2acc[concept] = {}

        for condition, condition_dict in concept_dict.items():

            condition_agreements = []
            for pair_idx, responses in condition_dict.items():
                agreement = responses.count(responses[0])/len(responses)
                if agreement < .5:
                    agreement = 1 - agreement
                condition_agreements.append(agreement)
                concept2pair2acc[concept][pair2stim[pair_idx]] = agreement

            mean_acc = np.mean(condition_agreements)
            concept2bucket2acc[concept][condition] = mean_acc
            concept2numraters[concept] = len(condition_agreements)

    return concept2bucket2acc, concept2pair2acc, concept2numraters

def plot_buckets(concept2bucket2acc, concept2numraters):

    df = {
        "concept": [],
        "bucket": [],
        "agreement": [],
    }
    for concept, bucket_dict in concept2bucket2acc.items():
        for bucket, agreement in bucket_dict.items():
            df["concept"].append(concept)
            df["bucket"].append(bucket)
            df["agreement"].append(agreement)

    data = pd.DataFrame.from_dict(df)
    sns.catplot(data=data, x="concept", y="agreement", hue="bucket", kind="bar")
    
    n_rater_str = ""
    for concept, n_raters in concept2numraters.items():
        n_rater_str += f"{concept}: {n_raters}, "
    plt.title(f"Mean P(Agree): {n_rater_str}")
    plt.ylim(0.5, 1.0)
    plt.savefig("buckets.png", bbox_inches="tight")

def plot_pairs(concept2pair2acc):

    df = {
        "stimulus": [],
        "agreement": [],
    }

    for concept, pair_dict in concept2pair2acc.items():
        df = {
            "stimulus": [],
            "agreement": [],
        }
         
        for stim, agreement in pair_dict.items():
            df["stimulus"].append(stim)
            df["agreement"].append(agreement)

        data = pd.DataFrame.from_dict(df)

        # Sort data by 'Value'
        data_sorted = data.sort_values(by='agreement', ascending=False)

        # Plot
        plt.figure()
        sns.barplot(data=data, x='stimulus', y='agreement', order=data_sorted['stimulus'])
        plt.xticks(rotation=90)
        plt.title( f"{concept} Agreement by Stimulus")
        plt.savefig(f"./{concept}_stim.png", bbox_inches="tight")



# Get all datafiles
root_dir = "../../data/Prolific/experiment_2_consistency_pilot"
fnames = os.listdir(root_dir)

concept_results = {
    "improbable_improbable": defaultdict(list),
    "impossible_impossible": defaultdict(list),
    "inconceivable_inconceivable": defaultdict(list),
    "improbable_impossible": defaultdict(list),
    "impossible_inconceivable": defaultdict(list),
}

results = {
    "improbable": copy.deepcopy(concept_results),
    "impossible": copy.deepcopy(concept_results),
    "inconceivable": copy.deepcopy(concept_results),
}

excluded_attn_count = 0

for fname in fnames:
    js = json.load(open(os.path.join(root_dir, fname), "rb"))
    json_data = json.loads(js["data"])
    critical_trials = [
        datum
        for datum in json_data
        if "task_type" in datum.keys() and datum["task_type"] == "critical"
    ]
    attn_trials = [
        datum
        for datum in json_data
        if "task_type" in datum.keys() and datum["task_type"] == "attention_check"
    ]


    if exclude_attention(attn_trials):
        excluded_attn_count += 1

    if not exclude_attention(attn_trials):
        results, pair2choices = process_critical_trials(critical_trials, results)

print(excluded_attn_count)

concept2bucket2acc, concept2pair2acc, concept2numraters = process_results(results, pair2choices)

plot_buckets(concept2bucket2acc, concept2numraters)
plot_pairs(concept2pair2acc)

