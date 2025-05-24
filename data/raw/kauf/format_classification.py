import os
import pandas as pd

data = pd.read_csv("./clean_DTFit_SentenceSet.csv")

formatted_data = {
    "item_set_id": [],
    "sentence": [],
    "label": [],
    "voice": [],
}

for row_id, row in data.iterrows():

    label = row["Plausibility"]
    if label == "Plausible":
        label = "probable"
    else:
        label = "improbable"

    formatted_data["item_set_id"].append(row["ItemNum"])
    formatted_data["sentence"].append(row["Sentence"])
    formatted_data["label"].append(label)
    formatted_data["voice"].append(row["Voice"])

os.makedirs("../../classification/kauf", exist_ok=True)
pd.DataFrame(formatted_data).to_csv("../../classification/kauf/DTFit_data.csv", index=False)

data = pd.read_csv("./clean_EventsRev_SentenceSet.csv")

formatted_data = {
    "item_set_id": [],
    "sentence": [],
    "label": [],
    "voice": [],
}

for row_id, row in data.iterrows():

    label = row["Plausibility"]
    if label == "Plausible":
        label = "probable"
    else:
        label = "improbable"

    formatted_data["item_set_id"].append(row["ItemNum"])
    formatted_data["sentence"].append(row["Sentence"])
    formatted_data["label"].append(label)
    formatted_data["voice"].append(row["Voice"])

os.makedirs("../../classification/kauf", exist_ok=True)
pd.DataFrame(formatted_data).to_csv("../../classification/kauf/EventsRev_data.csv", index=False)


data = pd.read_csv("./clean_EventsAdapt_SentenceSet.csv")
data = data[data["TrialType"] != "AAR"] # Drop reversible sentences

formatted_data = {
    "item_set_id": [],
    "sentence": [],
    "label": [],
    "voice": [],
    "synonym_set": [],
    "synonym_id": []
}

for row_id, row in data.iterrows():

    label = row["Plausibility"]
    condition = row["TrialType"]
    if label == "Plausible":
        label = "probable"
    elif condition == "AAN":
        label = "improbable"
    elif condition == "AI":
        label = "inconceivable"
    else:
        raise ValueError()

    formatted_data["item_set_id"].append(row["ItemNum"])
    formatted_data["sentence"].append(row["Sentence"])
    formatted_data["label"].append(label)
    formatted_data["voice"].append(row["Voice"])
    formatted_data["synonym_set"].append(row["SynonymPair"])
    formatted_data["synonym_id"].append(row["NumSyn"])

pd.DataFrame(formatted_data).to_csv("../../classification/kauf/EventsAdapt_data.csv", index=False)




