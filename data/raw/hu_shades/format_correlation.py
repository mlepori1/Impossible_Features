import os
import pandas as pd

data = pd.read_csv("./exp2_ratings.csv")

formatted_data = {
    "sentence": [],
    "Subjective Event Likelihood": [],
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
    
    # Get mean response
    mean_response = grp["response"].mean()
    formatted_data["sentence"].append(sentence)
    formatted_data["Subjective Event Likelihood"].append(mean_response)

os.makedirs("../../correlation/hu_shades", exist_ok=True)
pd.DataFrame(formatted_data).to_csv("../../correlation/hu_shades/data.csv", index=False)




