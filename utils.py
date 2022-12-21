import numpy as np
import pandas as pd
from rdkit import Chem as Chem
from rdkit.Chem import Draw
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

import umap

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
    score = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return score


def make_umap():
    """
    qm_descriptors and of the morgan fingerprints loading for umap
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

    # splitting into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    fps_train, fps_test, fps_y_train, fps_y_test = train_test_split(
        fps, y, test_size=0.2, random_state=42
    )

    indexes_d = np.where(y != 2)[0]
    indexes_train = np.where(fps_y_train != 2)[0]
    indexes_test = np.where(fps_y_test != 2)[0]

    # UMAP

    draw_umap(
        fps,
        y,
        title1="Umap of fingerprints -classes 0 and 1-",
        title2="Umap of fingerprints ",
        split=True,
    )

    draw_umap(
        X,
        y,
        title1="Umap of Molecular descr. -classes 0 and 1-",
        title2="Umap of Molecular descr. ",
        split=False,
    )


def draw_umap(
    data: np.array,
    targets: np.array,
    n_neighbors: int = 10,
    min_dist: float = 0.2,
    metric: str = "euclidean",
    title1: str = "",
    title2: str = "",
    split: bool = False,
):
    """creation of 2D UMAP and associated distribution plot"""

    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric=metric,
    )
    u = fit.fit_transform(data)

    if split == True:
        u, X_test, targets, y_test = train_test_split(
            u, targets, test_size=0.33, random_state=42
        )
        indexes_test = np.where(y_test != 2)[0]

    indexes = np.where(targets != 2)[0]

    u1 = np.concatenate(
        (
            u[indexes, :],
            np.resize(targets[indexes], (targets[indexes].shape[0], 1)),
        ),
        axis=1,
    )
    df = pd.DataFrame(u1, columns=["x", "y", "class"])
    sns.jointplot(df, x="x", y="y", hue="class")
    # if split == True:
    #    plt.title(title1 + ", train", y=1.3, fontsize=20)
    # else:
    #    plt.title(title1, y=1.3, fontsize=20)

    u2 = np.concatenate(
        (
            u,
            np.resize(targets, (targets.shape[0], 1)),
        ),
        axis=1,
    )
    df = pd.DataFrame(u2, columns=["x", "y", "class"])

    sns.jointplot(df, x="x", y="y", hue="class")
    # if split == True:
    #    plt.title(title2 + ", train", y=1.3,fontsize=20)
    # else:
    #    plt.title(title2, y=1.3, fontsize=20)

    if split == True:
        u3 = np.concatenate(
            (
                X_test[indexes_test, :],
                np.resize(
                    y_test[indexes_test],
                    (y_test[indexes_test].shape[0], 1),
                ),
            ),
            axis=1,
        )
        df = pd.DataFrame(u3, columns=["x", "y", "class"])
        sns.jointplot(df, x="x", y="y", hue="class")
        # plt.title(title1 + ", test", y=1.3,fontsize=20)

        u4 = np.concatenate(
            (
                X_test,
                np.resize(y_test, (y_test.shape[0], 1)),
            ),
            axis=1,
        )
        df = pd.DataFrame(u4, columns=["x", "y", "class"])

        sns.jointplot(df, x="x", y="y", hue="class")
        # plt.title(title2 + ", test", y=1.3, fontsize=20)


def draw_molecules(smiles, n=9):
    """
    Draw n molecules from the given dataset of smiles.
    :param smiles: list of SMILES
    :param n: number of molecules to draw
    :return: None
    """
    rnd_mol = [random.randint(0, len(smiles)) for i in range(n)]
    selected_mol = [Chem.MolFromSmiles(smiles[i]) for i in rnd_mol]

    # Plot
    fig = Draw.MolsToGridImage(
        selected_mol, molsPerRow=3, subImgSize=(800, 800)
    )
    fig.save("picture_molecule.png")
