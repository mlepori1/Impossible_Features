import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.metrics import mean_absolute_error, mean_squared_error
import scipy

import torch
import torch.nn.functional as F
import torch.optim as optim

import utils

# Expert label to human subjects' category options
nonsense_labels_mapping = {
    "near": "inconceivable",
    "far": "inconceivable",
    "probable": "probable",
}

goulding_labels_mapping = {
    "impossible": "impossible",
    "possible": "possible",
    "improbable": "possible",
}

shades_labels_mapping = {
    "impossible": "impossible",
    "probable": "probable",
    "improbable": "improbable",
    "inconceivable": "inconceivable",
}


def train_lr(features, labels, n_features, n_labels):
    # Helper function to train a logistic regression
    # classifier using soft labels
    torch.manual_seed(19)

    lr = torch.nn.Linear(n_features, n_labels).to("cuda")
    optimizer = optim.Adam(lr.parameters(), lr=0.01)

    epochs = 200

    # Convert to PyTorch tensors
    features = torch.tensor(features).to("cuda").float()
    labels = torch.tensor(labels).to("cuda").float()

    # Training LR
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = lr(features)
        log_probs = torch.log_softmax(logits, dim=1)
        # Loss is cross entropy with soft labels
        loss = -(labels * log_probs).sum(dim=1).mean()
        loss.backward()
        optimizer.step()
    return lr


def pred_lr(lr, features):
    # Use trained LR to get distributions
    with torch.no_grad():
        logits = lr(torch.tensor(features).to("cuda").float())
        probs = torch.softmax(logits, dim=1)
    return probs.cpu().numpy()


## Helper function to fit LR Models and evaluate them on held-out data
def lr_analysis(data, feature_names, dataset):

    # Define dataset level attributes
    if dataset == "hu_shades/data":
        mapping = shades_labels_mapping
    elif dataset == "hu_nonsense/data":
        mapping = nonsense_labels_mapping
    elif dataset == "goulding/adults":
        mapping = goulding_labels_mapping
    else:
        raise ValueError()

    class_labels = list(set(mapping.values()))
    class_labels.sort()

    # Map stimulus labels to human labels
    data["label"] = data["label"].map(mapping)

    # Store all results and summary stats
    results_df = {
        "KL Div": [],
        "MAE": [],
        "MSE": [],
        "Model Entropy": [],
        "Human Entropy": [],
        "Model Prob": [],
        "Human Prob": [],
        "Model Expert Prob": [],
        "Human Expert Prob": [],
        "Sentence": [],
    }

    # Compute metrics holding one stimulus out at a time
    for train_idxs, test_idxs in LeaveOneOut().split(data):

        train_data = data.iloc[train_idxs]
        test_data = data.iloc[test_idxs]

        # Ensure LOO is working correctly
        assert len(train_data) == len(data) - 1

        # Get soft labels from human classification
        soft_labels = train_data[class_labels].to_numpy()
        train_features = train_data[feature_names]

        # Fit LR with standardardized features
        scaler = StandardScaler().fit(train_features)
        train_features = scaler.transform(train_features)

        lr = train_lr(train_features, soft_labels, len(feature_names), len(class_labels))

        ### Compare the human-subjects classification behavior to LR predictions

        # Given trained LR, see how it does predicting the held out data point
        sample_features = scaler.transform(test_data[feature_names])

        # Predict probabilities for each class
        predicted_probabilities = pred_lr(lr, sample_features.reshape(1, -1))[0]

        # True probabilities drawn from human responses
        true_probabilites = np.array(
            [test_data.iloc[0][class_label] for class_label in class_labels]
        )

        # Asserts to ensure probabilities are computed correctly
        assert np.isclose(np.sum(true_probabilites), 1.0)
        assert np.isclose(np.sum(predicted_probabilities), 1.0)

        if dataset in ["hu_nonsense/data", "goulding/adults"]:
            assert len(predicted_probabilities) == 2
            assert len(true_probabilites) == 2
        else:
            assert len(predicted_probabilities) == 4
            assert len(true_probabilites) == 4

        # Compute Metrics
        kl_div = np.sum(
            scipy.special.rel_entr(true_probabilites, predicted_probabilities)
        )
        mae = mean_absolute_error(true_probabilites, predicted_probabilities)
        mse = mean_squared_error(true_probabilites, predicted_probabilities)

        results_df["KL Div"].append(kl_div)
        results_df["MAE"].append(mae)
        results_df["MSE"].append(mse)


        results_df["Model Entropy"].append(scipy.stats.entropy(predicted_probabilities))
        results_df["Human Entropy"].append(scipy.stats.entropy(true_probabilites))

        # For all datasets, pick out the first N-1 entries in the probability distribution.
        # These are independent, and will be flattened and correlated.
        results_df["Model Prob"].append(list(predicted_probabilities)[:-1])
        results_df["Human Prob"].append(list(true_probabilites)[:-1])

        expert_label = test_data.iloc[0]["label"]
        results_df["Model Expert Prob"].append(
            predicted_probabilities[class_labels.index(expert_label)]
        )
        results_df["Human Expert Prob"].append(
            true_probabilites[class_labels.index(expert_label)]
        )

        results_df["Sentence"].append(test_data.iloc[0]["sentence"])

    return results_df


def process_qualitative(data, condition, results_path):
    # Process results for Goulding dataset to give
    # qualitative results for paper
    qualitative_df = {
        "sentence": [],
        "model probability": [],
        "human probability": [],
    }

    data = pd.DataFrame().from_dict(data)
    for _, row in data.iterrows():
        qualitative_df["sentence"].append(row["Sentence"])
        qualitative_df["model probability"].append(1 - row["Model Prob"][0])
        qualitative_df["human probability"].append(1 - row["Human Prob"][0])

    pd.DataFrame().from_dict(qualitative_df).to_csv(
        os.path.join(results_path, "Qualitative_" + condition + ".csv")
    )

def generate_plot(model):
    # Generate plot of Shades of Zero data plotted with
    # background colored by logistic regression probabilities.
    # 
    # This plot is qualitative, to give the reader an idea of how 
    # the modal diff vector feature space partitions the data.
    # So just plot 3 classes along 2 axes

    path = os.path.join("../results", model, "Calibration", "hu_shades/data", "Linear_Representation.csv")
    data = pd.read_csv(path)

    feature_names = ["improbable_impossible", "impossible_inconceivable"]
    class_names = ["improbable", "impossible", "inconceivable"]

    # Sample just data belonging to these three classes
    data = data[data["label"].isin(class_names)]

    # Assign datapoints to colors by their expert label
    expert_labels = data["label"]

    colors = []
    for label in expert_labels:
        if label == "improbable":
            colors.append("blue")

        if label == "impossible":
            colors.append("gold")
        
        if label == "inconceivable":
            colors.append("red")

    # Generate soft labels, renormalizing to exclude "probable"
    soft_labels = data[class_names].to_numpy()
    soft_labels = soft_labels/np.sum(soft_labels, axis=1).reshape(-1, 1)

    # Sample just projections along 2 axes for plotting
    features = data[feature_names]

    # Standardize features
    features = StandardScaler().fit_transform(features)

    data = data.reset_index()        
    lr = train_lr(features, soft_labels, len(feature_names), 3)

    plt.Figure()
    _, ax = plt.subplots()

    legend_colors = {
        'Improbable': 'blue',
        'Impossible': 'gold',
        'Inconceivable': 'red'
    }

    # Create legend patches
    legend_handles = [mpatches.Patch(facecolor=color, label=label, edgecolor="black") for label, color in legend_colors.items()]

    ax.scatter(x=features[:, 0], y=features[:, 1], c=colors)
    plt.legend(handles=legend_handles, title="Expert Labels")

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 100),
        np.linspace(ylim[0], ylim[1], 100)
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]

    bg_probs = []
    for point_idx in range(len(grid_points)):
        grid_point = grid_points[point_idx]
        bg_probs.append(pred_lr(lr, grid_point.reshape(1, -1))[0])

    bg_probs = np.stack(bg_probs, axis=0)

    improbable_bg_probs = bg_probs[:, 0].reshape(xx.shape)
    impossible_bg_probs = bg_probs[:, 1].reshape(xx.shape)
    inconceivable_bg_probs = bg_probs[:, 2].reshape(xx.shape)

    # Loop over probability levels and draw a contour layer for each
    levels = np.linspace(0.01, 0.95, 20)
    for level in levels:
        ax.contourf(xx, yy, improbable_bg_probs,
                    levels=[level, 1],
                    colors='blue',
                    alpha=level/5,  # alpha increases with probability
                    antialiased=True)
        ax.contourf(xx, yy, inconceivable_bg_probs,
                    levels=[level, 1],
                    colors='red',
                    alpha=level/5,  # alpha increases with probability
                    antialiased=True)
        ax.contourf(xx, yy, impossible_bg_probs,
                    levels=[level, 1],
                    colors='yellow',
                    alpha=level/5,  # alpha increases with probability
                    antialiased=True)
    
    ax.scatter(x=features[:, 0], y=features[:, 1], c=colors, edgecolors="black")
    plt.title("Projections on Modal Difference Vectors")
    plt.ylabel("Impossible-Inconceivable")
    plt.xlabel("Improbable-Impossible")
    plt.savefig("../Figures/Study3_Example.pdf", format="pdf", bbox_inches="tight")


if __name__ == "__main__":

    config = utils.get_config()

    # Establish a dataframe to store all of the results
    # from the regression analysis
    regression_results = {
        "Model": [],
        "Dataset": [],
        "Condition": [],
        "MAE": [],
        "MSE": [],
        "KL": [],
        "Pearson R": [],
        "Entropy Correlation": [],
        "Expert Correlation": [],
    }

    # Map datasets to readable names
    dataset_map = {
        "hu_nonsense/data": "Hu et al. 2025a",
        "hu_shades/data": "Hu et al. 2025b",
        "goulding/adults": "Goulding",
    }

    model = config["model"]

    if model == "google/gemma-2-9b":
        generate_plot(model)

    for dataset in ["hu_nonsense/data", "hu_shades/data", "goulding/adults"]:
        print(dataset)
        for condition in ["Linear_Representation", "Probability", "PC", "Random"]:
            print(condition)
            if condition == "Probability":
                features = ["Probability"]
            else:
                features = [
                    "probable_improbable",
                    "improbable_impossible",
                    "impossible_inconceivable",
                ]

            path = os.path.join(
                "../results", model, "Calibration", dataset, f"{condition}.csv"
            )
            data = pd.read_csv(path)
            data = lr_analysis(data, features, dataset)

            # Store Calibration Results
            regression_results["Model"].append(model)
            regression_results["Dataset"].append(dataset_map[dataset])
            regression_results["Condition"].append(condition)
            regression_results["MAE"].append(np.mean(data["MAE"]))
            regression_results["MSE"].append(np.mean(data["MSE"]))
            regression_results["KL"].append(np.mean(data["KL Div"]))

            # Get N-1 probabilities from humans and models
            human_probs = np.concat(data["Human Prob"], axis=0)
            model_probs = np.concat(data["Model Prob"], axis=0)

            # Get overall distribution Correlations
            regression_results["Pearson R"].append(
                scipy.stats.pearsonr(human_probs, model_probs).statistic
            )
            # Get correlation between model and human entropy
            regression_results["Entropy Correlation"].append(
                scipy.stats.pearsonr(
                    data["Model Entropy"], data["Human Entropy"]
                ).statistic
            )
            # Get correlation between p(expert label) from models and humans
            regression_results["Expert Correlation"].append(
                scipy.stats.pearsonr(
                    data["Model Expert Prob"], data["Human Expert Prob"]
                ).statistic
            )

            # Save off examples for qualitative analysis
            if (
                model == "google/gemma-2-9b"
                and condition in ["Linear_Representation", "Probability"]
                and dataset == "goulding/adults"
            ):
                process_qualitative(data, condition, config["results_path"])

    regression_results = pd.DataFrame.from_dict(regression_results)

    regression_results.to_csv(
        os.path.join(config["results_path"], "regression_analysis.csv")
    )
