import os
import pandas as pd

data = pd.read_csv("./stimuli_with_syntax.csv")

formatted_data = {
    "item_set_id": [],
    "sentence": [],
    "label": [],
}

for row_id, row in data.iterrows():

    for label in ["probable", "improbable", "impossible", "inconceivable"]:
        prefix = row["sentence_prefix"]
        prefix = prefix.replace("[NP]", "Someone")
        prefix = prefix.replace("[POSS]", "their")
        suffix = row[label].replace("[POSS]", "their")
        sentence = prefix + " " + suffix + "."
        formatted_data["item_set_id"].append(row_id)
        formatted_data["sentence"].append(sentence)
        formatted_data["label"].append(label)

os.makedirs("../../classification/hu_shades", exist_ok=True)
pd.DataFrame(formatted_data).to_csv("../../classification/hu_shades/data.csv", index=False)




