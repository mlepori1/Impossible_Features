import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import json
import numpy as np
import copy


def compute_expectation(r_a, r_b):
    e_a = 1/(1 + (10 ** ((r_b - r_a)/400)))
    e_b = 1/(1 + (10 ** ((r_a - r_b)/400)))
    return e_a, e_b

def compute_update(rating, outcome, expectation, k=32):
    return rating + (k * (outcome - expectation))

def compute_ELO(stim2elo, trials):
    for trial in trials:
        stim0 = trial[0]
        stim1 = trial[1]
        winner = trial[2]

        r_0 = stim2elo[stim0]
        r_1 = stim2elo[stim1]

        e_0, e_1 = compute_expectation(r_0, r_1)
        r_0 = compute_update(r_0, winner, e_0)
        r_1 = compute_update(r_1, winner, e_1)

        stim2elo[stim0] = r_0
        stim2elo[stim1] = r_1

    return copy.deepcopy(stim2elo)

def reset_stim2elo(stim2elo):
    for k in stim2elo.keys():
        stim2elo[k] = 1400
    return stim2elo

def preprocess_trials(critical_trials):
    stim2elo = {}
    trials = []
    for trial in critical_trials:
        stim0 = trial["stimulus_0"] + " " + trial["continuation_0"] + "_" + trial["condition_0"]
        stim1 = trial["stimulus_1"] + " " + trial["continuation_1"]+ "_" + trial["condition_1"]
        winner = trial["response"]

        trials.append((stim0, stim1, winner))

        stim2elo[stim0] = 1400
        stim2elo[stim1] = 1400

    return stim2elo, trials



def estimate_ELO(critical_trials, N_ITERS=1000):
    elo_dicts = []
    stim2elo, trials = preprocess_trials(critical_trials)
    for _ in range(N_ITERS):
        stim2elo = reset_stim2elo(stim2elo)
        np.random.shuffle(trials)
        elo_dicts.append(compute_ELO(stim2elo, trials))
    
    elo_df = {
        "stimuli": [],
        "condition": [],
        "elo": [],
    }
    for stim in stim2elo.keys():
        elo = np.mean([elo_dict[stim] for elo_dict in elo_dicts])
        elo_df["stimuli"].append(stim)

        if "improbable" in stim:
            condition = "improbable"
        elif "impossible" in stim:
            condition = "impossible"
        elif "inconceivable" in stim:
            condition = "inconceivable"

        elo_df["condition"].append(condition)
        elo_df["elo"].append(elo)

    return pd.DataFrame.from_dict(elo_df)

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

def plot_ELO(df, outfile, title):
    plt.figure()
    sns.stripplot(data=df, y="elo", hue="condition")
    plt.title(title)
    plt.savefig(outfile)


# Get all datafiles
root_dir = "../../data/Prolific/Exp2_Pilot_ELO_Sample"
fnames = os.listdir(root_dir)

improb = []
imposs = []
inc = []

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
        if critical_trials[0]["concept"] == "improbable":
            improb += critical_trials
        elif critical_trials[0]["concept"] == "impossible":
            imposs += critical_trials
        elif critical_trials[0]["concept"] == "inconceivable":
            inc += critical_trials

improb = estimate_ELO(improb)
plot_ELO(improb, "./improbable_plot.png", "ELO on Improbable Axis")
improb.to_csv("./improbable.csv")

imposs = estimate_ELO(imposs)
plot_ELO(imposs, "./impossible_plot.png", "ELO on Impossible Axis")
imposs.to_csv("./impossible.csv")

inc = estimate_ELO(inc)
plot_ELO(inc, "./inconceivable_plot.png", "ELO on Inconceivable Axis")
inc.to_csv("./inconceivable.csv")

