import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import umap
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from shapely.geometry import Point
from shapely.ops import cascaded_union

####  TO CHECK THE LAST 2 FUNCTIONS
import umap.plot

from data_utils import *

from augmentation_utils import *
from conversion_smiles_utils import *


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


def make_umap():
    """
    plot of the umap of the qm_descriptors and of the morgan fingerprints
    """

    this_dir = os.getcwd()

    data_dir = os.path.join(this_dir, "data")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    ids, smiles, y = load_train_data(train_path)

    # descriptors

    X = smiles_to_qm_descriptors(smiles, data_dir)

    X, columns_info = nan_detection(
        X, smiles, nan_tolerance=0.0, cat_del=False
    )

    X, _ = transformation(
        X,
        smiles,
        columns_info,
        standardization=False,
        degree=1,
        pairs=False,
        log=False,
        fps=False,
    )  # nans are not accepted

    # fingerprints
    fps = smiles_to_morgan_fp(smiles)

    # p = np.random.permutation(len(y))
    # y = y[p]
    # X = X[p, :]

    # division in classe
    # class_indices = indices_by_class(y)
    # class_0 = X[class_indices[0]]
    # class_1 = X[class_indices[1]]
    # class_2 = X[class_indices[2]]

    # all_classes = np.concatenate([class_0, class_1, class_2], axis=0)

    # splitting into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    fps_train, fps_test, fps_y_train, fps_y_test = train_test_split(
        fps, y, test_size=0.33, random_state=42
    )

    indexes_d = np.where(y != 2)[0]
    indexes_train = np.where(fps_y_train != 2)[0]
    indexes_test = np.where(fps_y_test != 2)[0]

    # UMAP

    draw_umap(
        fps,
        y,
        n_components=2,
        title1="Umap of fingerprints -classes 0 and 1-",
        title2="Umap of fingerprints ",
        split=True,
    )

    draw_umap(
        X,
        y,
        n_components=2,
        title1="Umap of Molecular descr. -classes 0 and 1-",
        title2="Umap of Molecular descr. ",
        split=False,
    )


def draw_umap(
    data,
    targets,
    n_neighbors=10,
    min_dist=0.2,
    n_components=3,
    metric="euclidean",
    title1="",
    title2="",
    split=False,
):
    assert (
        n_components == 2 or n_components == 3
    ), f"n_components should 2 (2D) or 3 (3D): {n_components}"
    # https://umap-learn.readthedocs.io/en/latest/parameters.html

    colors = np.array(["red", "blue", "green"])

    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
    )
    u = fit.fit_transform(data)
    # fig = plt.figure()

    if split == True:
        u, X_test, targets, y_test = train_test_split(
            u, targets, test_size=0.33, random_state=42
        )
        indexes_test = np.where(y_test != 2)[0]

    indexes = np.where(targets != 2)[0]

    if n_components == 2:
        plt.figure(figsize=(7, 7))
        ax = plt.axes()
        ax.set_xlim(-10, 20)
        ax.set_ylim(-10, 20)
        fg = ax.scatter(
            u[indexes, 0],
            u[indexes, 1],
            c=colors[targets[indexes]],
            alpha=0.2,
            cmap="prism",
        )  # cmap="prism"
        if split == True:
            plt.title(title1 + ", train", fontsize=18)
        else:
            plt.title(title1, fontsize=18)

        plt.figure(figsize=(7, 7))
        ax = plt.axes()
        ax.set_xlim(-10, 20)
        ax.set_ylim(-10, 20)
        fg = ax.scatter(
            u[:, 0], u[:, 1], c=colors[targets], alpha=0.3, cmap="prism"
        )  # cmap="hsv"
        if split == True:
            plt.title(title2 + ", train", fontsize=18)
        else:
            plt.title(title2, fontsize=18)

        if split == True:
            plt.figure(figsize=(7, 7))
            ax = plt.axes()
            ax.set_xlim(-10, 20)
            ax.set_ylim(-10, 20)
            fg = ax.scatter(
                X_test[indexes_test, 0],
                X_test[indexes_test, 1],
                c=colors[y_test[indexes_test]],
                alpha=0.3,
                cmap="prism",
            )  # cmap="hsv"
            plt.title(title1 + ", test", fontsize=18)

            plt.figure(figsize=(7, 7))
            ax = plt.axes()
            ax.set_xlim(-10, 20)
            ax.set_ylim(-10, 20)
            fg = ax.scatter(
                X_test[:, 0],
                X_test[:, 1],
                c=colors[y_test],
                alpha=0.3,
                cmap="prism",
            )  # cmap="hsv"
            plt.title(title2 + ", test", fontsize=18)

    if n_components == 3:
        plt.figure(figsize=(7, 7))
        ax = plt.axes(projection="3d")
        ax.set_xlim3d(-10, 20)
        ax.set_ylim3d(-10, 20)
        ax.set_zlim3d(-10, 20)
        fg = ax.scatter3D(
            u[indexes, 0],
            u[indexes, 1],
            u[indexes, 2],
            c=colors[targets[indexes]],
            alpha=0.3,
            cmap="prism",
        )

        if split == True:
            plt.title(title1 + ", train", fontsize=18)
        else:
            plt.title(title1, fontsize=18)

        plt.figure(figsize=(7, 7))
        ax = plt.axes(projection="3d")
        ax.set_xlim3d(-10, 20)
        ax.set_ylim3d(-10, 20)
        ax.set_zlim3d(-10, 20)
        fg = ax.scatter3D(
            u[:, 0],
            u[:, 1],
            u[:, 2],
            c=colors[targets],
            alpha=0.3,
            cmap="prism",
        )
        if split == True:
            plt.title(title2 + ", train", fontsize=18)
        else:
            plt.title(title2, fontsize=18)

        if split == True:
            plt.figure(figsize=(7, 7))
            ax = plt.axes(projection="3d")
            ax.set_xlim3d(-10, 20)
            ax.set_ylim3d(-10, 20)
            ax.set_zlim3d(-10, 20)
            fg = ax.scatter3D(
                X_test[indexes_test, 0],
                X_test[indexes_test, 1],
                X_test[indexes_test, 2],
                c=colors[y_test[indexes_test]],
                alpha=0.3,
                cmap="prism",
            )
            plt.title(title1 + ", test", fontsize=18)

            plt.figure(figsize=(7, 7))
            ax = plt.axes(projection="3d")
            ax.set_xlim3d(-10, 20)
            ax.set_ylim3d(-10, 20)
            ax.set_zlim3d(-10, 20)
            fg = ax.scatter3D(
                X_test[:, 0],
                X_test[:, 1],
                X_test[:, 2],
                c=colors[y_test],
                alpha=0.3,
                cmap="prism",
            )  # cmap="hsv"
            plt.title(title2 + ", test", fontsize=18)
