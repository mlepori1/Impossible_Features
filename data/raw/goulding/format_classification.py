import os
import pandas as pd

data = pd.read_csv("./Exp_1_data.csv")
triad_groups = data.groupby("Triad No.")

formatted_data = {
    "item_set_id": [],
    "sentence": [],
    "label": [],
}

for triad_id, grp in triad_groups:
    events = grp["Event"].unique()
    for event in events:
        label = grp[grp["Event"] == event]["event_type"].iloc[0]
        if label == "ordinary": label = "probable"
        
        sentence  = event[:-1] # strip "?""
        sentence = "Someone is about to " + sentence + "."
        formatted_data["item_set_id"].append(triad_id)
        formatted_data["sentence"].append(sentence)
        formatted_data["label"].append(label)

os.makedirs("../../classification/goulding", exist_ok=True)
pd.DataFrame(formatted_data).to_csv("../../classification/goulding/data.csv", index=False)




