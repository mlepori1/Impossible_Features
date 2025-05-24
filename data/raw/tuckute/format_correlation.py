import os
import pandas as pd

data = pd.read_csv("./drive_suppress_data.csv")

formatted_data = {
    "sentence": [],
    "5-Gram Log-Prob": [],
    "GPT2-XL Log-Prob": [],
    "PCFG Log-Prob": [],
    "Arousal": [],
    "Conversational": [],
    "Sense": [],
    "Gram.": [],
    "Frequency": [],
    "Imageability": [],
    "Others Thoughts": [],
    "Physical": [],
    "Places": [],
    "Valence": [],
}

data = data.groupby("sentence")
for sentence, grp in data:

    row = grp.iloc[0]
    formatted_data["sentence"].append(sentence)
    formatted_data["5-Gram Log-Prob"].append(row["log-prob-5gram_mean"])
    formatted_data["GPT2-XL Log-Prob"].append(row["log-prob-gpt2-xl_mean"])
    formatted_data["PCFG Log-Prob"].append(row["log-prob-pcfg_mean"])
    formatted_data["Arousal"].append(row["rating_arousal_mean"])
    formatted_data["Conversational"].append(row["rating_conversational_mean"])
    formatted_data["Sense"].append(row["rating_sense_mean"])
    formatted_data["Gram."].append(row["rating_gram_mean"])
    formatted_data["Frequency"].append(row["rating_frequency_mean"])
    formatted_data["Imageability"].append(row["rating_imageability_mean"])
    formatted_data["Others Thoughts"].append(row["rating_others_thoughts_mean"])
    formatted_data["Physical"].append(row["rating_physical_mean"])
    formatted_data["Places"].append(row["rating_places_mean"])
    formatted_data["Valence"].append(row["rating_valence_mean"]),

os.makedirs("../../correlation/tuckute", exist_ok=True)
pd.DataFrame(formatted_data).to_csv("../../correlation/tuckute/data.csv", index=False)




