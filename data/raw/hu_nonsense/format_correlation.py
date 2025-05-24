import os
import pandas as pd
import numpy as np
from collections import defaultdict

data = pd.read_csv("./ranking.csv")

formatted_data = {
    "sentence": [],
    "Ranked Inconceivability": [],
}

sentences = data["ranked_sentences"].iloc[0]
ids = data["ranked_unique_ids"].iloc[0]

id2sentence = {}

sentences = sentences[1:-1].split(",")
ids = ids[1:-1].split(",")

for idx in range(len(sentences)):
    sentence = sentences[idx].strip()[1:-1] + "."
    id = int(ids[idx])
    id2sentence[id] = sentence


ids2rankings = defaultdict(list)

for _, row in data.iterrows():

    ids = row["ranked_unique_ids"]
    ids = ids[1:-1].split(",")
    ids = [int(id) for id in ids]

    for rank in range(len(ids)):
        ids2rankings[ids[rank]].append(rank)

for id in ids2rankings.keys():
    mean_ranking = np.mean(ids2rankings[id]).item()
    sentence = id2sentence[id]

    formatted_data["sentence"].append(sentence)
    formatted_data["Ranked Inconceivability"].append(mean_ranking)


os.makedirs("../../correlation/hu_nonsense", exist_ok=True)
pd.DataFrame(formatted_data).to_csv("../../correlation/hu_nonsense/data.csv", index=False)




