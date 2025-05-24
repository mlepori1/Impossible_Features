"""This script formats the datasets from Vega-Mendoza et al 2021 into a standard CSV format
"""

import json
import pandas as pd

df = {
    "sentence_0": [],
    "sentence_1": [],
    "modal_0": [],
    "modal_1": [],
    "semantics_0": [],
    "semantics_1": [],
}

file = open("Not_Quite_Sherlock_Datasets/eng__better_likely__worse_impossible__worse_related.json")
data = json.load(file)
for datum in data:
    df["sentence_0"].append(datum["Better"])
    df["sentence_1"].append(datum["Worse"])
    df["modal_0"].append("probable")
    df["modal_1"].append("impossible")
    df["semantics_0"].append("related")
    df["semantics_1"].append("related")


file = open("Not_Quite_Sherlock_Datasets/eng__better_likely__worse_impossible__worse_unrelated.json")
data = json.load(file)
for datum in data:
    df["sentence_0"].append(datum["Better"])
    df["sentence_1"].append(datum["Worse"])
    df["modal_0"].append("probable")
    df["modal_1"].append("impossible")
    df["semantics_0"].append("related")
    df["semantics_1"].append("unrelated")

file = open("Not_Quite_Sherlock_Datasets/eng__better_likely__worse_possible__worse_related.json")
data = json.load(file)
for datum in data:
    df["sentence_0"].append(datum["Better"])
    df["sentence_1"].append(datum["Worse"])
    df["modal_0"].append("probable")
    df["modal_1"].append("improbable")
    df["semantics_0"].append("related")
    df["semantics_1"].append("related")


file = open("Not_Quite_Sherlock_Datasets/eng__better_likely__worse_possible__worse_unrelated.json")
data = json.load(file)
for datum in data:
    df["sentence_0"].append(datum["Better"])
    df["sentence_1"].append(datum["Worse"])
    df["modal_0"].append("probable")
    df["modal_1"].append("improbable")
    df["semantics_0"].append("related")
    df["semantics_1"].append("unrelated")

file = open("Not_Quite_Sherlock_Datasets/eng__better_unlikely__better_related__worse_related.json")
data = json.load(file)
for datum in data:
    df["sentence_0"].append(datum["Better"])
    df["sentence_1"].append(datum["Worse"])
    df["modal_0"].append("improbable")
    df["modal_1"].append("impossible")
    df["semantics_0"].append("related")
    df["semantics_1"].append("related")


file = open("Not_Quite_Sherlock_Datasets/eng__better_unlikely__better_related__worse_unrelated.json")
data = json.load(file)
for datum in data:
    df["sentence_0"].append(datum["Better"])
    df["sentence_1"].append(datum["Worse"])
    df["modal_0"].append("improbable")
    df["modal_1"].append("impossible")
    df["semantics_0"].append("related")
    df["semantics_1"].append("unrelated")

file = open("Not_Quite_Sherlock_Datasets/eng__better_unlikely__better_unrelated__worse_related.json")
data = json.load(file)
for datum in data:
    df["sentence_0"].append(datum["Better"])
    df["sentence_1"].append(datum["Worse"])
    df["modal_0"].append("improbable")
    df["modal_1"].append("impossible")
    df["semantics_0"].append("unrelated")
    df["semantics_1"].append("related")


file = open("Not_Quite_Sherlock_Datasets/eng__better_unlikely__better_unrelated__worse_unrelated.json")
data = json.load(file)
for datum in data:
    df["sentence_0"].append(datum["Better"])
    df["sentence_1"].append(datum["Worse"])
    df["modal_0"].append("improbable")
    df["modal_1"].append("impossible")
    df["semantics_0"].append("unrelated")
    df["semantics_1"].append("unrelated")

pd.DataFrame.from_dict(df).to_csv("./sherlock.csv", index=False)