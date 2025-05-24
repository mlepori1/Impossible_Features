import os
import pandas as pd
import numpy as np

data = pd.read_csv("./exp1_trial_data.csv")

formatted_data = {
    "sentence": [],
    "label": [],
    "probable": [],
    "improbable": [],
    "impossible": [],
    "inconceivable": [],
}

data = data.groupby("unique_id")
for _, grp in data:

    # Format sentence
    row = grp.iloc[0]
    condition = row["condition"]
    prefix = row["sentence_prefix"]
    prefix = prefix.replace("[NP]", "Someone")
    prefix = prefix.replace("[POSS]", "their")
    suffix = row[condition].replace("[POSS]", "their")
    sentence = prefix + " " + suffix + "."
    
    # Get proportion of respondents for each class
    probable = np.sum(grp["response_label"] == "probable")/len(grp)
    improbable = np.sum(grp["response_label"] == "improbable")/len(grp)
    impossible = np.sum(grp["response_label"] == "impossible")/len(grp)
    inconceivable = np.sum(grp["response_label"] == "nonsense")/len(grp)

    formatted_data["sentence"].append(sentence)
    formatted_data["label"].append(condition)
    formatted_data["probable"].append(probable)
    formatted_data["improbable"].append(improbable)
    formatted_data["impossible"].append(impossible)
    formatted_data["inconceivable"].append(inconceivable)

os.makedirs("../../calibration/hu_shades", exist_ok=True)
pd.DataFrame(formatted_data).to_csv("../../calibration/hu_shades/data.csv", index=False)




