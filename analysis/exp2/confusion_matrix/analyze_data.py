import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix


class TieBreakerKNN(KNeighborsClassifier):

    def _get_neighbors(self, distances, indices, k):
        selected_indices = []
        for i in range(distances.shape[0]):
            unique_distances = np.unique(distances[i])
            selected = []
            for dist in unique_distances:
                tied_points = indices[i][distances[i] == dist]
                np.random.shuffle(tied_points)
                remainder = k - len(selected)
                selected.extend(tied_points[:remainder])

                if len(selected) == k:
                    break
            selected_indices.append(selected)
        return np.array(selected_indices)

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        all_neighbors = self._fit_X.shape[0]
        distances, indices = super().kneighbors(
            X, n_neighbors=all_neighbors, return_distance=True
        )
        indices = self._get_neighbors(distances, indices, self.n_neighbors)
        return (distances, indices) if return_distance else indices

def process_critical_trials(critical_trials):
    print("Process Critical")

    prompt_type = critical_trials[0]["prompt"]
    probable_projections = [
        datum["response"]
        for datum in critical_trials
        if datum["condition"] == "probable"
    ]
    improbable_projections = [
        datum["response"]
        for datum in critical_trials
        if datum["condition"] == "improbable"
    ]
    impossible_projections = [
        datum["response"]
        for datum in critical_trials
        if datum["condition"] == "impossible"
    ]
    inconceivable_projections = [
        datum["response"]
        for datum in critical_trials
        if datum["condition"] == "inconceivable"
    ]

    x = np.array(
        probable_projections
        + improbable_projections
        + impossible_projections
        + inconceivable_projections
    )
    y = np.array(
        [0] * len(probable_projections)
        + [1] * len(improbable_projections)
        + [2] * len(impossible_projections)
        + [3] * len(inconceivable_projections)
    )

    knn = TieBreakerKNN(n_neighbors=3)

    # Save Preds for confusion_matrix
    y_gt = []
    y_pred = []

    # Leave-one-out CV
    cv = LeaveOneOut()
    for train_idxs, test_idx in cv.split(x):
        x_train, x_test = x[train_idxs], x[test_idx]
        y_train, y_test = y[train_idxs], y[test_idx]

        knn.fit(x_train.reshape(-1, 1), y_train)
        y_pred.append(knn.predict(x_test.reshape(-1, 1))[0])
        y_gt.append(y_test[0])

    confusion = confusion_matrix(y_gt, y_pred, normalize="true")
    return confusion, prompt_type


def exclude_critical(critical_trials):
    # If any critical trial had a probable sentence rated at 75\% improbable, impossible or incoceivable, exclude
    for trial in critical_trials:
        response = trial["response"]
        probable_bool = trial["condition"] == "probable"
        if probable_bool and response > 75:
            return True
    return False


def plot_results(improb, imposs, inc):

    labels = ["probable", "improbable", "impossible", "inconceivable"]

    improb = np.mean(np.stack(improb, axis=0), axis=0)
    imposs = np.mean(np.stack(imposs, axis=0), axis=0)
    inc = np.mean(np.stack(inc, axis=0), axis=0)
    print(inc)

    # Plot results
    plt.figure()
    g = sns.heatmap(
        data=improb,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        cmap=sns.color_palette("light:#5A9", as_cmap=True),
    )
    plt.ylabel("Correct Class")
    plt.xlabel("Predicted Class")
    plt.title(f"Improbable Confusion Matrix")
    plt.savefig("./improbable.png")
    plt.close()

    plt.figure()
    g = sns.heatmap(
        data=imposs,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        cmap=sns.color_palette("light:#5A9", as_cmap=True),
    )
    plt.ylabel("Correct Class")
    plt.xlabel("Predicted Class")
    plt.title(f"Impossible Confusion Matrix")
    plt.savefig("./impossible.png")
    plt.close()

    plt.figure()
    g = sns.heatmap(
        data=inc,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        cmap=sns.color_palette("light:#5A9", as_cmap=True),
    )
    plt.ylabel("Correct Class")
    plt.xlabel("Predicted Class")
    plt.title(f"Inconceivable Confusion Matrix")
    plt.savefig("./inconceivable.png")
    plt.close()


# Get all datafiles
root_dir = "../../data/Prolific/Exp2_Pilot"
fnames = os.listdir(root_dir)

improb = []
imposs = []
inc = []

excluded_critical_count = 0

for fname in fnames:
    js = json.load(open(os.path.join(root_dir, fname), "rb"))
    json_data = json.loads(js["data"])
    critical_trials = [
        datum
        for datum in json_data
        if "task_type" in datum.keys() and datum["task_type"] == "critical"
    ]

    if exclude_critical(critical_trials):
        excluded_critical_count += 1

    if not exclude_critical(critical_trials):
        cm, prompt = process_critical_trials(critical_trials)
        if prompt == "improbable":
            improb.append(cm)
        elif prompt == "impossible":
            imposs.append(cm)
        elif prompt == "inconceivable":
            inc.append(cm)
    else:
        print(js["prolific_data"]["subject_id"])

print(excluded_critical_count)
plot_results(improb, imposs, inc)
