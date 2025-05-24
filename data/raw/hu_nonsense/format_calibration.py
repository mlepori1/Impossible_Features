import os
import pandas as pd
import numpy as np

data = pd.read_csv("./norming_data.csv")
MIN_RESPONSES = 4

formatted_data = {
    "sentence": [],
    "label": [],
    "probable": [],
    "inconceivable": [],
}

data = data.groupby("sentence")
for sentence, grp in data:

    if len(grp) < MIN_RESPONSES:
        continue

    # Format sentence
    row = grp.iloc[0]
    condition = row["condition"]
    if condition == "target":
        condition = "probable"
    elif condition == "near":
        condition = "near"
    else:
        condition = "far"
    sentence = sentence.capitalize() + "."
    
    # Get proportion of respondents for each class
    probable = np.sum(grp["response_label"] == "not_total_nonsense")/len(grp)
    inconceivable = np.sum(grp["response_label"] == "total_nonsense")/len(grp)

    formatted_data["sentence"].append(sentence)
    formatted_data["label"].append(condition)
    formatted_data["probable"].append(probable)
    formatted_data["inconceivable"].append(inconceivable)

os.makedirs("../../calibration/hu_nonsense", exist_ok=True)
pd.DataFrame(formatted_data).to_csv("../../calibration/hu_nonsense/data.csv", index=False)




