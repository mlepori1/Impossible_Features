import pandas as pd
import random

import torch
from torch.utils.data import Dataset


class SentenceFeaturesDataset(Dataset):

    def __init__(self, file="stimuli_with_syntax"):
        self.data = pd.read_csv(f"{file}.csv")

        self.base = []
        self.probable = []
        self.improbable = []
        self.impossible = []
        self.inconceivable = []

        for i, row in self.data.iterrows():
            # If continuation contains [POSS], just replace with "their" as a possessive pronoun
            base = row["classification_prefix"].replace("[POSS]", "their")
            self.base.append(base)

            self.probable.append(
                "They are " + base + " " + row["probable"].replace("[POSS]", "their") + "."
            )
            self.improbable.append(
                "They are " + base + " " + row["improbable"].replace("[POSS]", "their") + "."
            )
            self.impossible.append(
                "They are " + base + " " + row["impossible"].replace("[POSS]", "their") + "."
            )
            self.inconceivable.append(
                "They are " + base + " " + row["inconceivable"].replace("[POSS]", "their") + "."
            )


    def __len__(self):
        return len(self.probable)

    def __getitem__(self, idx):
        return {
            "base": self.base[idx],
            "probable": self.probable[idx],
            "improbable": self.improbable[idx],
            "impossible": self.impossible[idx],
            "inconceivable": self.inconceivable[idx],
        }
