import pandas as pd
import random

import torch
from torch.utils.data import Dataset


class SentenceFeaturesDataset(Dataset):

    def __init__(self, file="stimuli_with_syntax"):
        self.data = pd.read_csv(f"../data/{file}.csv")

        self.base = []
        self.probable = []
        self.improbable = []
        self.impossible = []
        self.inconceivable = []
        self.syntactic = []
        self.shuffled = []
        self.monsters = []
        self.birds = []

        for i, row in self.data.iterrows():
            # If continuation contains [POSS], just replace with "their" as a possessive pronoun
            base = row["classification_prefix"].replace("[POSS]", "their")
            self.base.append(base)

            self.probable.append(
                base + " " + row["probable"].replace("[POSS]", "their") + "."
            )
            self.improbable.append(
                base + " " + row["improbable"].replace("[POSS]", "their") + "."
            )
            self.impossible.append(
                base + " " + row["impossible"].replace("[POSS]", "their") + "."
            )
            self.inconceivable.append(
                base + " " + row["inconceivable"].replace("[POSS]", "their") + "."
            )
            self.syntactic.append(
                base
                + " "
                + row["inconceivable_syntactic"].replace("[POSS]", "their")
                + "."
            )

            to_shuffle = base + row["probable"].replace("[POSS]", "their")
            to_shuffle = to_shuffle.split(" ")
            random.shuffle(to_shuffle)
            shuffled = " ".join(to_shuffle)
            self.shuffled.append(shuffled + ".")

    def __len__(self):
        return len(self.probable)

    def __getitem__(self, idx):
        return {
            "base": self.base[idx],
            "probable": self.probable[idx],
            "improbable": self.improbable[idx],
            "impossible": self.impossible[idx],
            "inconceivable": self.inconceivable[idx],
            "syntactic": self.syntactic[idx],
            "shuffled": self.shuffled[idx],
        }
