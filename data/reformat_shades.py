"""Reformat Shades dataset for projections"""

import pandas as pd

data = pd.read_csv("shades_of_zero.csv")

df = {"sentence": [], "class": []}

for i, row in data.iterrows():
    base = row["classification_prefix"].replace("[POSS]", "their")
    df["sentence"].append(base + " " + row["probable"].replace("[POSS]", "their") + ".")
    df["class"].append("probable")

    df["sentence"].append(
        base + " " + row["improbable"].replace("[POSS]", "their") + "."
    )
    df["class"].append("improbable")

    df["sentence"].append(
        base + " " + row["impossible"].replace("[POSS]", "their") + "."
    )
    df["class"].append("impossible")

    df["sentence"].append(
        base + " " + row["inconceivable"].replace("[POSS]", "their") + "."
    )
    df["class"].append("inconceivable")

pd.DataFrame.from_dict(df).to_csv("shades_reformated.csv", index=False)
