from augmentation_utils import *
from conversion_smiles_utils import *

import os
import csv
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import math
import scipy as sc
from typing import Tuple, List

import rdkit
from rdkit import Chem as Chem
from rdkit.Chem import AllChem as AllChem
from rdkit.Chem import rdFingerprintGenerator

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA
from sklearn.utils import resample

from mordred import Calculator, descriptors
from imblearn.over_sampling import SMOTE
import h5py


def load_train_data(train_path: str):

    df = pd.read_csv(train_path)

    smiles = df["smiles"].values.tolist()
    targets = df["sol_category"].values.tolist()
    targets = np.array(targets)

    if "Id" in df.columns:
        ids = df["Id"].values.tolist()
        return ids, smiles, targets
    else:
        return smiles, targets


def load_test_data(test_path: str) -> Tuple[List[str], List[str]]:

    df = pd.read_csv(test_path)

    ids = df["Id"].values.tolist()
    smiles = df["smiles"].values.tolist()

    return ids, smiles

def create_submission_file(
    ids: List[str], y_pred: np.array, path: str
) -> None:
    """
    Function to create the final submission file
    Args:
        ids: list of ids corresponding to the original data
        y_pred: np.array of model predictions
        path: path to where the csv should be saved

    Returns:
        None
    """
    with open(path, "w") as submission_file:
        writer = csv.writer(submission_file, delimiter=",")
        writer.writerow(["Id", "Pred"])
        for id, pred in zip(ids, y_pred):
            writer.writerow([id, pred])


def up_down_sampling(y, X):
    """
    Add copies of the observations of the minority classes and remove observations from the majority class
    Args:
        y: classes values
        X: observations

    Returns:
        y_balanced
        X_balanced
    """

    # dividing the data in subsets depending on the class
    X2 = X[np.where(y == 2)]
    X1 = X[np.where(y == 1)]
    X0 = X[np.where(y == 0)]

    # up-sample minority classes
    X0_up = resample(
        X0, replace=True, n_samples=int(X.shape[0] * 0.25), random_state=13
    )
    X1_up = resample(
        X1, replace=True, n_samples=int(X.shape[0] * 0.25), random_state=13
    )

    # down-sample majority class
    X2_down = resample(
        X2, replace=False, n_samples=int(X.shape[0] * 0.5), random_state=13
    )

    X_balanced = np.concatenate((X0_up, X1_up, X2_down))
    y_balanced = np.concatenate(
        (
            np.zeros(X0_up.shape[0], dtype=np.int8),
            np.ones(X1_up.shape[0], dtype=np.int8),
            2 * np.ones(X2_down.shape[0], dtype=np.int8),
        )
    )
    print(X_balanced.shape, y_balanced.shape)

    return y_balanced, X_balanced


def smote_algorithm(y, X, seed: int):
    """
    Up-sample the minority class
    Args:
        y: classes values
        X: observations

    Returns:
        y_resampled
        X_resampled
    """
    sm = SMOTE(random_state=seed)
    X_resampled, y_resampled = sm.fit_resample(X, y)

    return y_resampled, X_resampled


def calculate_class_weights(
    targets: np.array, num_classes: int = 3
) -> List[float]:
    """
    computation of the weights to eal with the umbalanceness of the dataset
    """

    # see how balanced the data is and assign weights
    train_data_size = targets.shape[0]

    weights = [
        1 - np.count_nonzero(targets == int(i)) / train_data_size
        for i in range(num_classes)
    ]

    return weights


def indices_by_class(
    targets: np.array, num_classes: int = 3
) -> List[np.array]:
    """
    returns the indices divided following the 3 different solubility classes
    """
    class_indices = []
    for class_ in range(num_classes):
        class_idx = np.where(targets == class_)[0]
        class_indices.append(class_idx)
    # print('Call to the function indices_by_class returns: ')
    # print(class_indices)
    return class_indices


def split_by_class(targets: np.array, CV: int = 5, num_classes: int = 3):
    splitted_classes_indices = indices_by_class(targets, num_classes)

    kfold = KFold(n_splits=CV)

    train_indices = [[[] for i in range(num_classes)] for j in range(CV)]
    test_indices = [[[] for i in range(num_classes)] for j in range(CV)]
    for idx, class_i_indices in enumerate(splitted_classes_indices):
        for split_i, (train_split_i, test_split_i) in enumerate(
            kfold.split(class_i_indices)
        ):
            train_idx_class_i = class_i_indices[train_split_i]
            train_indices[split_i][idx].append(train_idx_class_i)

            test_idx_class_i = class_i_indices[test_split_i]
            test_indices[split_i][idx].append(test_idx_class_i)

    train_indices = [
        np.concatenate(i, axis=1).squeeze() for i in train_indices
    ]
    test_indices = [np.concatenate(i, axis=1).squeeze() for i in test_indices]

    final_splits = [
        (train_i, test_i)
        for train_i, test_i in zip(train_indices, test_indices)
    ]

    return final_splits


def create_subsample_train_csv(data_dir: str, features: np.array):
    path = os.path.join(data_dir, "random_undersampling.csv")
    train_path = os.path.join(data_dir, "train.csv")
    ids, smiles, targets = load_train_data(train_path)

    p = np.random.permutation(targets.shape[0])
    smiles = np.array(smiles)[p]
    targets = targets[p]
    features = features[p]

    # dividing the data in subsets depending on the class
    smiles_per_class = []
    all_features = []
    for class_idx in range(3):
        indices = np.where(targets == class_idx)
        smiles_class_i = np.array(smiles)[indices].tolist()
        smiles_per_class.append(smiles_class_i)

        features_class_i = features[indices]
        all_features.append(features_class_i)

    min_len = min([len(i) for i in smiles_per_class])

    cutoff_features = []
    with open(path, "w") as subsampling_file:
        writer = csv.writer(subsampling_file, delimiter=",")
        writer.writerow(["Id", "smiles", "sol_category"])

        for idx, smiles_class_i in enumerate(smiles_per_class):
            subsampled_features = all_features[idx][:min_len]
            cutoff_features.append(subsampled_features)

            subsampled_smiles = smiles_class_i[:min_len]
            for temp_smiles in subsampled_smiles:
                writer.writerow(["-", temp_smiles, idx])

    cutoff_features = np.concatenate(cutoff_features, axis=0)
    descriptor_file = os.path.join(
        data_dir, "random_undersampling_descriptors.h5"
    )
    with h5py.File(descriptor_file, "w") as hf:
        hf.create_dataset("descriptors", data=cutoff_features)

"""
def PCA_application(dataset, targets, dataset_test,targets_test):

    X = pd.DataFrame(dataset)
    y = pd.DataFrame(targets)
    
    X_output = pd.DataFrame(dataset_test)
    y_output = pd.DataFrame(targets_test)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    std = StandardScaler()
    X_train_std = std.fit_transform(X_train)
    X_test_std = std.transform(X_test)
    X_output_std = std.transform(X_output)

    pca = PCA(n_components=X_train_std.shape[1])
    pca_data = pca.fit_transform(X_train_std)

    percent_var_explained = pca.explained_variance_ / (
        np.sum(pca.explained_variance_)
    )
    cumm_var_explained = np.cumsum(percent_var_explained)

    plt.plot(cumm_var_explained)
    plt.grid()
    plt.xlabel("n_components")
    plt.ylabel("% variance explained")
    plt.show()

    cum = cumm_var_explained
    var = pca.explained_variance_
    value = pca.explained_variance_ratio_
    index = np.where(cum >= value)[0][0]

    pca = PCA(n_components=index)
    pca_train_data = pca.fit_transform(X_train_std)
    pca_test_data = pca.transform(X_test_std)
    pca_output_data = pca.transform(X_output_std)
    return pca_train_data,pca_test_data,pca_output_data
"""


def PCA_application(dataset, dataset_test):
    # https://www.youtube.com/watch?v=oiusrJ0btwA
    X_train = pd.DataFrame(dataset)

    X_output = pd.DataFrame(dataset_test)

    pca = PCA(n_components=X_train.shape[1])
    pca_data = pca.fit_transform(X_train)
    # pca_output_data = pca.transform(X_output)

    explained_variance = pca.explained_variance_ratio_
    cumm_var_explained = np.cumsum(explained_variance)

    plt.plot(cumm_var_explained)
    plt.grid()
    plt.xlabel("n_components")
    plt.ylabel("% variance explained")
    plt.show()

    cum = cumm_var_explained
    var = pca.explained_variance_
    index = np.where(cum >= 0.975)[0][0]

    pca = PCA(n_components=index + 1)
    pca_train_data = pca.fit_transform(X_train)
    pca_output_data = pca.transform(X_output)
    print(pca_train_data)
    print(type(pca_train_data))
    print(type(pca_output_data))
    # print(pca_train_data)
    return (
        pca_train_data,
        pca_output_data,
    )  # pca_train_data.to_numpy(), pca_output_data.to_numpy()


if __name__ == "__main__":

    this_dir = os.getcwd()

    data_dir = os.path.join(this_dir, "data")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    ids, smiles, targets = load_train_data(train_path)

    qm_descriptors = smiles_to_qm_descriptors(smiles, data_dir)
    (
        dataset,
        columns_info,
    ) = nan_imputation(qm_descriptors, 0.0, standardization=False)

    submission_ids, submission_smiles = load_test_data(test_path)

    # TEST SET TRANSFORMATION
    # descriptors
    qm_descriptors_test = smiles_to_qm_descriptors(
        submission_smiles, data_dir, "test"
    )

    # UMAP

    p = np.random.permutation(len(targets))
    targets_shuffled = targets[p]
    dataset_shuffled = dataset[p, :]
    # make_umap(
    #    dataset_shuffled[0 : int(math.modf(len(targets) * 0.6)[1]), :],
    #    targets_shuffled[0 : int(math.modf(len(targets) * 0.6)[1])],
    # )
    # make_umap(dataset_shuffled, targets_shuffled)
    submission_ids, submission_smiles = load_test_data(test_path)
    """
    make_umap(
        dataset_shuffled[int(math.modf(len(targets) * 0.6)[1]) : -1, :],
        targets_shuffled[int(math.modf(len(targets) * 0.6)[1]) : -1],
    )
    """


