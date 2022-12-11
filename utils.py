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

    X, _, _ = nan_imputation(
        X,
        smiles,
        nan_tolerance=0.0,
        standardization=False,
        cat_del=False,
        log=False,
    )  # nans are not accepted

    # fingerprints
    fps = smiles_to_morgan_fp(smiles)

    p = np.random.permutation(len(y))
    y = y[p]
    X = X[p, :]

    # division in classe
    class_indices = indices_by_class(y)
    class_0 = X[class_indices[0]]
    class_1 = X[class_indices[1]]
    class_2 = X[class_indices[2]]

    all_classes = np.concatenate([class_0, class_1, class_2], axis=0)

    # splitting into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    fps_train, fps_test, fps_y_train, fps_y_test = train_test_split(
        fps, y, test_size=0.33, random_state=42
    )

    indexes_train = np.where(fps_y_train != 2)[0]
    indexes_test = np.where(fps_y_test != 2)[0]

    # UMAP
    draw_umap(
        fps_train[indexes_train, :],
        fps_y_train[indexes_train],
        n_components=2,
        title="Umap of Morgan fingerprints - train, classes 0 and 1 -",
    )
    draw_umap(
        fps_X_test[indexes_test, :],
        fps_y_test[indexes_test],
        n_components=2,
        title="Umap of fingerprints - test, classes 0 and 1 -",
    )

    draw_umap(
        fps_X_train,
        fps_y_train,
        n_components=2,
        title="Umap of fingerprints - train -",
    )
    draw_umap(
        fps_X_test,
        fps_y_test,
        n_components=2,
        title="Umap of fingerprints - test -",
    )


def draw_umap(
    data,
    targets,
    n_neighbors=15,
    min_dist=0.1,
    n_components=3,
    metric="euclidean",
    title="",
):
    assert (
        n_components == 2 or n_components == 3
    ), f"n_components should 2 (2D) or 3 (3D): {n_components}"
    # https://umap-learn.readthedocs.io/en/latest/parameters.html
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
    )
    u = fit.fit_transform(data)
    fig = plt.figure()

    if n_components == 2:
        plt.figure(figsize=(7, 7))
        ax = plt.axes()
        ax.set_xlim(-10, 15)
        ax.set_ylim(-10, 15)
        fg = ax.scatter(
            u[:, 0], u[:, 1], c=targets, alpha=0.1, cmap="prism"
        )  # cmap="hsv"
    if n_components == 3:
        plt.figure(figsize=(7, 7))
        ax = plt.axes(projection="3d")
        ax.set_xlim3d(-10, 15)
        ax.set_ylim3d(-10, 15)
        ax.set_zlim3d(-10, 15)
        fg = ax.scatter3D(
            u[:, 0], u[:, 1], u[:, 2], c=targets, alpha=0.1, cmap="prism"
        )

    plt.title(title, fontsize=18)


"""
def scatter_umap(u, class_0, class_1, class_2):

    alpha = 0.5
    polygons1 = [
        Point(u[i, 0], u[i, 1]).buffer(0.08) for i in range(len(class_0))
    ]
    polygons2 = [
        Point(u[len(class_0) + i, 0], u[len(class_0) + i, 1]).buffer(0.08)
        for i in range(len(class_1))
    ]
    polygons3 = [
        Point(u[len(class_1) + i, 0], u[len(class_1) + i, 1]).buffer(0.08)
        for i in range(len(class_2))
    ]
    polygons1 = cascaded_union(polygons1)
    polygons2 = cascaded_union(polygons2)
    polygons3 = cascaded_union(polygons3)
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, title="Umap_points")
    for polygon1 in polygons1:
        polygon1 = ptc.Polygon(
            np.array(polygon1.exterior), facecolor="red", lw=0, alpha=alpha
        )
        ax.add_patch(polygon1)
    for polygon2 in polygons2:
        polygon2 = ptc.Polygon(
            np.array(polygon2.exterior), facecolor="blue", lw=0, alpha=alpha
        )
        ax.add_patch(polygon2)
    for polygon3 in polygons3:
        polygon3 = ptc.Polygon(
            np.array(polygon3.exterior), facecolor="green", lw=0, alpha=alpha
        )
        ax.add_patch(polygon3)
    ax.axis([-5, 15, -5, 15])


def scatter_umap_3d(u, class_0, class_1, class_2):

    alpha = 0.5
    polygons1 = [
        Point(u[i, 0], u[i, 1], u[i, 2]).buffer(0.06)
        for i in range(len(class_0))
    ]
    polygons2 = [
        Point(
            u[len(class_0) + i, 0],
            u[len(class_0) + i, 1],
            u[len(class_0) + i, 2],
        ).buffer(0.06)
        for i in range(len(class_1))
    ]
    polygons3 = [
        Point(
            u[len(class_1) + i, 0],
            u[len(class_1) + i, 1],
            u[len(class_1) + i, 2],
        ).buffer(0.06)
        for i in range(len(class_2))
    ]
    polygons1 = cascaded_union(polygons1)
    polygons2 = cascaded_union(polygons2)
    polygons3 = cascaded_union(polygons3)
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, title="Umap_points")
    for polygon1 in polygons1:
        polygon1 = ptc.Polygon(
            np.array(polygon1.exterior), facecolor="red", lw=0, alpha=alpha
        )
        ax.add_patch(polygon1)
    for polygon2 in polygons2:
        polygon2 = ptc.Polygon(
            np.array(polygon2.exterior), facecolor="blue", lw=0, alpha=alpha
        )
        ax.add_patch(polygon2)
    for polygon3 in polygons3:
        polygon3 = ptc.Polygon(
            np.array(polygon3.exterior), facecolor="green", lw=0, alpha=alpha
        )
        ax.add_patch(polygon3)
    ax.axis([-5, 15, -5, 15])
"""
