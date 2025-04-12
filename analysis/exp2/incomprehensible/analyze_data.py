import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas
import json
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
def process_critical_trials(critical_trials):
    print("Process Critical")
    # Create a dict mapping stimulus to response for each axis
    improbable_data = [datum for datum in critical_trials if datum["prompt"] == "improbable"]
    impossible_data = [datum for datum in critical_trials if datum["prompt"] == "impossible"]
    inconceivable_data = [datum for datum in critical_trials if datum["prompt"] == "inconceivable"]

    improbable_dict = {datum["sentence_prefix"] + "_" + datum["condition"]: datum["response"] for datum in improbable_data}
    impossible_dict = {datum["sentence_prefix"] + "_" + datum["condition"]: datum["response"] for datum in impossible_data}
    inconceivable_dict = {datum["sentence_prefix"] + "_" + datum["condition"]: datum["response"] for datum in inconceivable_data}

    # Keys are the same for all dicts, so create a vector where indices are matched to stimuli
    keys = improbable_dict.keys()
    improbable_vector = []
    impossible_vector = []
    inconceivable_vector = []
    for key in keys:
        improbable_vector.append(improbable_dict[key])
        impossible_vector.append(impossible_dict[key])
        inconceivable_vector.append(inconceivable_dict[key])
    
    # calculate correlations
    improb_imposs = pearsonr(improbable_vector, impossible_vector).statistic
    improb_inc = pearsonr(improbable_vector, inconceivable_vector).statistic
    imposs_inc = pearsonr(impossible_vector, inconceivable_vector).statistic

    return np.array([
        [1.0, improb_imposs, improb_inc],
        [improb_imposs, 1.0, imposs_inc],
        [improb_inc, imposs_inc, 1.0]
    ])

def exclude_attention(attention_checks):
    # assess attention checks
    attn_results = []
    for idx, attention_check in enumerate(attention_checks):
        response = attention_check["choices"][attention_check["response"]]
        if "improbable" in response and "improbable" in attention_check["correct_answer"]:
            attn_results.append(True)
        elif "impossible" in response and "impossible" in attention_check["correct_answer"]:
            attn_results.append(True)
        elif "incomprehensible" in response and "incomprehensible" in attention_check["correct_answer"]:
            attn_results.append(True)
        else:
            attn_results.append(False)

    if not np.all(attn_results):
        return True
    else:
        return False

def exclude_critical(critical_trials):
    # If any critical trial had a probable sentence rated at 75\% improbable, impossible, incoceivable, exclude
    for trial in critical_trials:
        response = trial["response"]
        probable_bool = trial["condition"] == "probable"
        if probable_bool and response > 75:
            return True
    return False

def plot_results(data):
    # Plot individual results
    for idx, map in enumerate(data):
        plt.figure()
        sns.heatmap(data=map, cbar=True)
        plt.title(f"Subject: {idx} Correlations")
        plt.savefig(f"./subject_{idx}.png")

    # Plot average results
    avg_map = np.mean(np.stack(data), axis=0)
    plt.figure()
    sns.heatmap(data=avg_map, cbar=True)
    plt.title(f"Averaged Correlations")
    plt.savefig(f"./correlations.png")
    
# Get all datafiles
root_dir = "../../data/Prolific/Exp2_Pilot"
fnames = os.listdir(root_dir)

data = []
excluded_attention_count = 0
excluded_critical_count = 0

for fname in fnames:
    js = json.load(open(os.path.join(root_dir, fname), "rb"))
    json_data = json.loads(js["data"])
    attention_checks = [datum for datum in json_data if "task_type" in datum.keys() and datum["task_type"] == "attention_check"]
    critical_trials = [datum for datum in json_data if "task_type" in datum.keys() and datum["task_type"] == "critical"]

    if exclude_attention(attention_checks):
        excluded_attention_count += 1

    if exclude_critical(critical_trials):
        excluded_critical_count += 1

    if not exclude_attention(attention_checks) and not exclude_critical(critical_trials):
        data.append(process_critical_trials(critical_trials))
    else:
        print(js["prolific_data"]["subject_id"])

print(excluded_critical_count)
print(excluded_attention_count)
plot_results(data)

