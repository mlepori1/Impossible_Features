import os
import pandas as pd
import json
from collections import defaultdict
import numpy as np
files = os.listdir(".")

stim2responses = defaultdict(list)
for f in files:
    if ".json" in f:
        data = json.loads(json.load(open(f, "rb"))["data"])
        critical_trials = [
            datum
            for datum in data
            if "task_type" in datum.keys() and datum["task_type"] == "critical"
        ]

        responses = []
        for trial in critical_trials:
            responses.append(trial["response"])
        
        mean = np.mean(responses)
        std = np.std(responses)
        for trial in critical_trials:
            stimuli = trial["classification_prefix"] + " " + trial[trial["condition"]] + "."
            response = (trial["response"] - mean)/std
            stim2responses[stimuli].append(response)

df = {
    "sentence": [],
    "imageability": [],
}

for stim, responses in stim2responses.items():
    df["sentence"].append(stim)
    df["imageability"].append(np.mean(responses))

data = pd.DataFrame.from_dict(df)
data.to_csv("visualization_data.csv")