import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import umap
import plotly.express as px
from sklearn.decomposition import PCA

# import umap.plot

from data_preparation import *


def quadratic_weighted_kappa(y_pred: np.array, y_true: np.array) -> float:
    """
    Function to compute the quadratic weighted kappa which is the primary metric for the competition
    Args:
        y_pred: np.array containing the model predictions
        y_true: np.array containing the true target values

    Returns:

    """
    # following https://www.kaggle.com/code/aroraaman/quadratic-kappa-metric-explained-in-5-simple-steps/notebook
    score = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return score


def differential_top_k_loss(y_pred, y_true):
    # following https://github.com/Felix-Petersen/difftopk
    pass


"""
def make_umap(X: np.array, y: np.array):
    X, _ = nan_imputation(
        X, nan_tolerance=0.0, standardization=False, cat_del=False
    )  # nans are not accepted

    class_indices = indices_by_class(y)
    class_0 = X[class_indices[0]]
    class_1 = X[class_indices[1]]
    class_2 = X[class_indices[2]]

    all_classes = np.concatenate(
        [class_0, class_1, class_2], axis=0
    )  # X is sufficient

    # could append all fingerprints together -> make sure you know split them based on the label
    # reducer = umap.UMAP(n_components=3)
    reducer = umap.UMAP(n_neighbors=5)
    mapper = reducer.fit(all_classes)

    umap.plot.points(mapper)

    # plot all 3 labels
    red = umap.UMAP(n_components=3, n_neighbors=5)
    embedding = red.fit_transform(all_classes)
    print(embedding)
    plt.figure()

    umap.plot.diagnostic(mapper, diagnostic_type="pca")
    plt.figure()

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(
        embedding[: len(class_0), 0],
        embedding[: len(class_0), 1],
        embedding[: len(class_0), 2],
        edgecolor="red",
        linewidths=0.5,
        label="0",
    )
    ax.scatter(
        embedding[len(class_0) : len(class_0) + len(class_1), 0],
        embedding[len(class_0) : len(class_0) + len(class_1), 1],
        embedding[len(class_0) : len(class_0) + len(class_1), 2],
        edgecolor="blue",
        linewidths=0.5,
        label="1",
    )
    ax.scatter(
        embedding[len(class_0) + len(class_1) :, 0],
        embedding[len(class_0) + len(class_1) :, 1],
        embedding[len(class_0) + len(class_1) :, 2],
        edgecolor="yellow",
        linewidths=0.5,
        label="2",
    )

    plt.show()

    plt.scatter(
        embedding[: len(class_0), 0],
        embedding[: len(class_0), 1],
        embedding[: len(class_0), 2],
        edgecolor="red",
        linewidths=0.5,
        label="0",
    )
    plt.show()
    plt.scatter(
        embedding[len(class_0) : len(class_0) + len(class_1), 0],
        embedding[len(class_0) : len(class_0) + len(class_1), 1],
        embedding[len(class_0) : len(class_0) + len(class_1), 2],
        edgecolor="blue",
        linewidths=0.5,
        label="1",
    )
    plt.show()
    plt.scatter(
        embedding[len(class_0) + len(class_1) :, 0],
        embedding[len(class_0) + len(class_1) :, 1],
        embedding[len(class_0) + len(class_1) :, 2],
        edgecolor="yellow",
        linewidths=0.5,
        label="2",
    )
    plt.show()
"""
