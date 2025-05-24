import os
import pandas as pd
import numpy as np

data = pd.read_csv("./Exp_1_data.csv")
age_5 = data[data["AgeGroup"] == "age 5"]
age_7 = data[data["AgeGroup"] == "age 7"]
age_9 = data[data["AgeGroup"] == "age 9"]
adult = data[data["AgeGroup"] == "adults"]

def format_age(data, filename):

    grouped_data = data.groupby("Event")

    formatted_data = {
        "sentence": [],
        "label": [],
        "possible": [],
        "impossible": [],
    }

    for event, grp in grouped_data:
        
        row = grp.iloc[0]
        condition = row["event_type"]
        if condition in ["ordinary"]:
            condition = "possible"
        
        sentence  = event[:-1] # strip "?""
        sentence = "Someone is about to " + sentence + "."
        possible = np.sum(grp["Response"] == 1)/ len(grp)
        impossible = np.sum(grp["Response"] == 0)/ len(grp)

        formatted_data["sentence"].append(sentence)
        formatted_data["label"].append(condition)
        formatted_data["possible"].append(possible)
        formatted_data["impossible"].append(impossible)

    os.makedirs("../../calibration/goulding", exist_ok=True)
    pd.DataFrame(formatted_data).to_csv(f"../../calibration/goulding/{filename}.csv", index=False)

format_age(age_5, "age_5")
format_age(age_7, "age_7")
format_age(age_9, "age_9")
format_age(adult, "adults")