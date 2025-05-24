import os
import json
import pandas as pd

formatted_data = {
    "item_set_id": [],
    "sentence": [],
    "label": [],
    "metadata": [],
}

file = open("./eng__better_likely__worse_impossible__worse_related.json")
data = json.load(file)
for datum in data:
    formatted_data["item_set_id"].append(datum["Item"])
    formatted_data["sentence"].append(datum["Better"])
    formatted_data["label"].append("probable")
    formatted_data["metadata"].append("")

    formatted_data["item_set_id"].append(datum["Item"])
    formatted_data["sentence"].append(datum["Worse"])
    formatted_data["label"].append("inconceivable")
    formatted_data["metadata"].append("related")


file = open("./eng__better_likely__worse_impossible__worse_unrelated.json")
data = json.load(file)
for datum in data:
    formatted_data["item_set_id"].append(datum["Item"])
    formatted_data["sentence"].append(datum["Better"])
    formatted_data["label"].append("probable")
    formatted_data["metadata"].append("")

    formatted_data["item_set_id"].append(datum["Item"])
    formatted_data["sentence"].append(datum["Worse"])
    formatted_data["label"].append("inconceivable")
    formatted_data["metadata"].append("unrelated")

file = open("./eng__better_likely__worse_possible__worse_related.json")
data = json.load(file)
for datum in data:
    formatted_data["item_set_id"].append(datum["Item"])
    formatted_data["sentence"].append(datum["Better"])
    formatted_data["label"].append("probable")
    formatted_data["metadata"].append("")

    formatted_data["item_set_id"].append(datum["Item"])
    formatted_data["sentence"].append(datum["Worse"])
    formatted_data["label"].append("improbable")
    formatted_data["metadata"].append("related")


file = open("./eng__better_likely__worse_possible__worse_unrelated.json")
data = json.load(file)
for datum in data:
    formatted_data["item_set_id"].append(datum["Item"])
    formatted_data["sentence"].append(datum["Better"])
    formatted_data["label"].append("probable")
    formatted_data["metadata"].append("")

    formatted_data["item_set_id"].append(datum["Item"])
    formatted_data["sentence"].append(datum["Worse"])
    formatted_data["label"].append("improbable")
    formatted_data["metadata"].append("unrelated")

file = open("./eng__better_unlikely__better_related__worse_related.json")
data = json.load(file)
for datum in data:
    formatted_data["item_set_id"].append(datum["Item"])
    formatted_data["sentence"].append(datum["Better"])
    formatted_data["label"].append("improbable")
    formatted_data["metadata"].append("related")

    formatted_data["item_set_id"].append(datum["Item"])
    formatted_data["sentence"].append(datum["Worse"])
    formatted_data["label"].append("inconceivable")
    formatted_data["metadata"].append("related")


file = open("./eng__better_unlikely__better_related__worse_unrelated.json")
data = json.load(file)
for datum in data:
    formatted_data["item_set_id"].append(datum["Item"])
    formatted_data["sentence"].append(datum["Better"])
    formatted_data["label"].append("improbable")
    formatted_data["metadata"].append("related")

    formatted_data["item_set_id"].append(datum["Item"])
    formatted_data["sentence"].append(datum["Worse"])
    formatted_data["label"].append("inconceivable")
    formatted_data["metadata"].append("unrelated")

file = open("./eng__better_unlikely__better_unrelated__worse_related.json")
data = json.load(file)
for datum in data:
    formatted_data["item_set_id"].append(datum["Item"])
    formatted_data["sentence"].append(datum["Better"])
    formatted_data["label"].append("improbable")
    formatted_data["metadata"].append("unrelated")

    formatted_data["item_set_id"].append(datum["Item"])
    formatted_data["sentence"].append(datum["Worse"])
    formatted_data["label"].append("inconceivable")
    formatted_data["metadata"].append("related")


file = open("./eng__better_unlikely__better_unrelated__worse_unrelated.json")
data = json.load(file)
for datum in data:
    formatted_data["item_set_id"].append(datum["Item"])
    formatted_data["sentence"].append(datum["Better"])
    formatted_data["label"].append("improbable")
    formatted_data["metadata"].append("unrelated")

    formatted_data["item_set_id"].append(datum["Item"])
    formatted_data["sentence"].append(datum["Worse"])
    formatted_data["label"].append("inconceivable")
    formatted_data["metadata"].append("unrelated")

os.makedirs("../../classification/vega_mendoza", exist_ok=True)
data = pd.DataFrame(formatted_data)
data = data.sort_values(by="item_set_id", ignore_index=True)
data = data.drop_duplicates()
data.to_csv("../../classification/vega_mendoza/data.csv", index=False)