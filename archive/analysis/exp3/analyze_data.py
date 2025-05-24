import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas
import json
import numpy as np
from sklearn.metrics import cohen_kappa_score
import itertools

def get_accuracy(data, majority=False):
    if majority == False:
        accs = np.concatenate(data, axis=0)
        return np.mean(accs)
    if majority == True:
        gt = np.stack(data, axis=0)
        gt = np.mean(data, axis=0) > 0.5
        print(gt)
        accs = []

        for dataset in data:
            for idx in range(len(dataset)):
                accs.append(dataset[idx] == gt[idx])
        return np.mean(accs)

def process_critical_trials(critical_trials):
    # First, compute accuracy on the noncontroversial stimuli
    improb = [datum for datum in critical_trials if datum["class"] == "improbable"]
    impossible = [datum for datum in critical_trials if datum["class"] == "impossible"]
    controversial = [datum for datum in critical_trials if datum["class"] == "controversial"]

    acc_improb = [1 if datum["response_label"] == "improbable" else 0 for datum in improb]
    acc_imposs = [1 if datum["response_label"] == "impossible" else 0 for datum in impossible]
    responses_controversial = [1 if datum["response_label"] == "improbable" else 0 for datum in controversial]

    return acc_improb, acc_imposs, responses_controversial

def plot_results(improb, imposs, controversial):
    # Plot interrater agreement for all categories
    kappa_improb = get_accuracy(improb)
    kappa_imposs = get_accuracy(imposs)
    kappa_controversial = get_accuracy(controversial, majority=True)

    print([kappa_improb, kappa_controversial, kappa_imposs])
    plt.bar(x=range(3), height=[kappa_improb, kappa_controversial, kappa_imposs])
    plt.xlabel(["improbable", "controversial", "impossible"])
    plt.title("Average Accuracy over Subjects")
    plt.savefig("agreement.png")

# Get all datafiles
root_dir = "../../data/Prolific/Exp3a_Pilot"
fnames = os.listdir(root_dir)

improb_responses = []
imposs_responses = []
controversial_responses = []

for fname in fnames:
    js = json.load(open(os.path.join(root_dir, fname), "rb"))
    json_data = json.loads(js["data"])
    critical_trials = [datum for datum in json_data if "task_type" in datum.keys() and datum["task_type"] == "critical"]

    results = process_critical_trials(critical_trials)
    improb_responses.append(results[0])
    imposs_responses.append(results[1])
    controversial_responses.append(results[2])

plot_results(improb_responses, imposs_responses, controversial_responses)
