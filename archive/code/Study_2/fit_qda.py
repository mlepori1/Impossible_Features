"""
This file fits a QDA on a training set and evaluates
it on a test set, providing the posterior probability
of each class for each test set examples
"""

import argparse
import os
import pickle as pkl

import pandas as pd
import numpy as np

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

VECTORS = [
    "probable_improbable",
    "improbable_impossible",
    "impossible_inconceivable",
]


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        default="gemma-2-2b",
        help="Model to use as backbone for analysis",
    )

    parser.add_argument(
        "--outfolder",
        default="./Study_2/results/",
        type=str,
        help="Folder to put results",
    )

    parser.add_argument(
        "--train_dataset",
        default="projected_shades_reformated.csv",
        type=str,
        help="Data file to fit QDA on",
    )

    parser.add_argument(
        "--eval_dataset",
        default="projected_shades_reformated.csv",
        type=str,
        help="Data file to compute QDA probabilities on",
    )

    parser.add_argument(
        "--features",
        default=["improbable_impossible", "impossible_inconceivable"],
        type=list,
        help="Features to include in the QDA",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Parse Args
    args = parse_arguments()

    ### Set Up Output
    outfolder = os.path.join(args.outfolder, args.model)
    os.makedirs(outfolder, exist_ok=True)


    ### Load Dataset
    train_dataset = pd.read_csv(os.path.join(outfolder, args.train_dataset))
    train_dataset = train_dataset[train_dataset["class"] != "probable"]
    train_features = train_dataset[args.features]
    train_classes = train_dataset["class"]

    eval_dataset = pd.read_csv(os.path.join(outfolder, args.eval_dataset))
    eval_dataset = eval_dataset[eval_dataset["class"] != "probable"]

    eval_features = eval_dataset[args.features]
    eval_classes = eval_dataset["class"]

    qda = QuadraticDiscriminantAnalysis()
    qda.fit(train_features, train_classes)

    classes = qda.classes_
    probs = qda.predict_proba(eval_features)

    df = {
        "sentence": [],
        "improbable": [],
        "impossible": [],
        "inconceivable": [],
    }

    for stim_idx, dist in enumerate(probs):
        df["sentence"].append(eval_dataset["sentence"].iloc[stim_idx])
        for cl_idx, cl in enumerate(classes):
            df[cl].append(dist[cl_idx])

    dataname = args.eval_dataset.split("/")[-1]
    pd.DataFrame.from_dict(df).to_csv(os.path.join(outfolder, f"QDA_Projections_{dataname}"), index=False)
    pkl.dump(qda, open(os.path.join(outfolder, f"{dataname[:-4]}_qda.pkl"), "wb"))
