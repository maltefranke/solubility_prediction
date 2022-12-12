import os
import csv
from typing import Tuple, List
import math
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from augmentation_utils import *
from utils import *
from data_utils import *
from conversion_smiles_utils import *
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
)
from matplotlib.colors import ListedColormap
import seaborn as sns
import warnings

# TO BE CORRECTED
warnings.filterwarnings("ignore")
# predictions_svm_class_constantlr_0.01_l1 submission (2) su leo->0.01283
def svm_learning(
    X: np.array,
    y: np.array,
    CV: int = 5,
    label_weights: List[float] = None,
    seed: int = 13,
    sample_weights: List[float] = None,
):  # -----------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=123
    )
    sample_weights = [label_weights[i] for i in y_train]
    label_weights = {
        0: label_weights[0],
        1: label_weights[1],
        2: label_weights[2],
    }  # -----------------------------------------------------

    learning_rate = ["constant", "optimal", "invscaling"]
    eta0 = [0.1, 0.05, 0.01]
    penalty = ["l1", "l2"]
    random_grid = {
        "learning_rate": learning_rate,
        "eta0": eta0,
        "penalty": penalty,
    }

    kappa_scorer = make_scorer(quadratic_weighted_kappa)
    clf = SGDClassifier(class_weight=label_weights)
    gs = RandomizedSearchCV(
        clf,
        param_distributions=random_grid,
        n_iter=10,
        cv=5,
        verbose=2,
        random_state=seed,
        n_jobs=-1,
        scoring=kappa_scorer,
    )
    # -----------------------------------------------------
    # Train model
    gs.fit(X_train, y_train, sample_weight=sample_weights)
    print("---------------results------------------------------")

    print("The best parameters are %s" % (gs.best_params_))
    # Predict on test set
    y_pred = gs.best_estimator_.predict(X_test)
    # Get Probability estimates
    # y_prob = gs.best_estimator_.predict_proba(X_test)[:, 1]
    # -----------------------------------------------------
    print("Best_score: %.2f%%" % (gs.best_score))
    print("Accuracy score: %.2f%%" % (accuracy_score(y_test, y_pred) * 100))
    print(
        "Precision score: %.2f%%"
        % (precision_score(y_test, y_pred, average="weighted") * 100)
    )
    print(
        "Recall score: %.2f%%"
        % (recall_score(y_test, y_pred, average="weighted") * 100)
    )
    # -----------------------------------------------------
    # Plot confusion matrix
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5))
    cm = confusion_matrix(y_test, y_pred)  # , labels= target_names)
    sns.heatmap(
        cm,
        annot=True,
        cbar=False,
        fmt="d",
        linewidths=0.5,
        cmap="Blues",
        ax=ax1,
    )
    ax1.set_title("Confusion Matrix")
    ax1.set_xlabel("Predicted class")
    ax1.set_ylabel("Actual class")
    ax1.set_xticklabels(target_names)
    ax1.set_yticklabels(target_names)
    fig.tight_layout()
    # -----------------------------------------------------
    # Plot the decision boundary
    cmap = ListedColormap(colors[: len(np.unique(y_test))])
    # plot the decision surface
    x1_min, x1_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    x2_min, x2_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    resolution = 0.01  # step size in the mesh
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution),
    )
    Z = gs.best_estimator_.predict(np.c_[xx1.ravel(), xx2.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    # plot class samples
    for idx, cl in enumerate(np.unique(y_test)):
        plt.scatter(
            x=X_test[y_test == cl, 0],
            y=X_test[y_test == cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=target_names[cl],
            edgecolor="black",
        )
    ax2.set_title(title)
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":

    this_dir = os.path.dirname(os.getcwd())

    data_dir = os.path.join(
        this_dir, "solubility_prediction\data"
    )  # MODIFY depending on your folder!!
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    # get data and transform smiles
    ids, smiles, targets = load_train_data(train_path)

    # DATASET TRANSFORMATION

    # degree = (
    #    2  # -> to be put in "preprocessing()" if you want power augmentation
    # )
    dataset, columns_info, log_trans = preprocessing(ids, smiles, data_dir)
    train_data_size = targets.shape[0]

    # we permute/shuffle our data first
    seed = 13
    np.random.seed(seed)
    p = np.random.permutation(targets.shape[0])
    dataset = dataset[p]
    targets = targets[p]

    # TEST SET

    submission_ids, submission_smiles = load_test_data(test_path)

    # TEST SET TRANSFORMATION
    # descriptors
    qm_descriptors_test = smiles_to_qm_descriptors(
        submission_smiles, data_dir, "test"
    )

    qm_descriptors_test, _ = transformation(
        qm_descriptors_test,
        columns_info,
        standardization=True,
        test=True,
        degree=1,
        pairs=False,
        log_trans=log_trans,
    )

    weights = calculate_class_weights(targets)
    sample_weights = [weights[i] for i in targets]

    svm_learning(
        dataset,
        targets,
        CV=5,
        label_weights=weights,
        sample_weights=sample_weights,
        seed=seed,
    )
