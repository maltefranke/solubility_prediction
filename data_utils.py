import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List

from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.utils import resample

from imblearn.over_sampling import SMOTE
import h5py

from conversion_smiles_utils import *


def load_train_data(train_path: str) -> Tuple[List[str], List[str], np.array]:
    """
    Load the data from the .csv file with the train dataset
    Args:
        train_path: path to the csv file containing the data

    Returns:
        list of ids, list of SMILES, np.array of target values
    """

    df = pd.read_csv(train_path)

    smiles = df["smiles"].values.tolist()
    targets = df["sol_category"].values.tolist()
    targets = np.array(targets)

    ids = df["Id"].values.tolist()
    return ids, smiles, targets


def load_test_data(test_path: str) -> Tuple[List[str], List[str]]:
    """
    Load the test data from the .csv file
    Args:
        test_path: path to the csv file containing the data

    Returns:
        list of ids, list of SMILES
    """

    df = pd.read_csv(test_path)

    ids = df["Id"].values.tolist()
    smiles = df["smiles"].values.tolist()

    return ids, smiles


def create_submission_file(ids: List[str], y_pred: np.array, path: str) -> None:
    """
    Function to create the final submission file
    Args:
        ids: list of ids corresponding to the original data
        y_pred: array of model predictions'
        path: path to where the csv should be saved

    Returns:
        None
    """
    with open(path, "w") as submission_file:
        writer = csv.writer(submission_file, delimiter=",")
        writer.writerow(["Id", "Pred"])
        for id, pred in zip(ids, y_pred):
            writer.writerow([id, pred])


def calculate_class_weights(y: np.array, num_classes: int = 3) -> List[float]:
    """
    Computation of the weights to deal with imbalance of the dataset
    Args:
        y: np.array of target values
        num_classes: number of classes in the classification task

    Returns:
        list of class weights
    """
    train_data_size = y.shape[0]

    weights = [
        1 - np.count_nonzero(y == int(i)) / train_data_size
        for i in range(num_classes)
    ]

    return weights


def up_down_sampling(y: np.array, X: np.array) -> Tuple[np.array, np.array]:
    """
    Add copies of the observations of the minority classes and remove observations from the majority class
    Args:
        y: np.array of target values
        X: observations

    Returns:
        balanced y and X
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


def smote_algorithm(y: np.array, X: np.array, seed: int) -> Tuple[np.array, np.array]:
    """
    Up-sample the minority class
    Args:
        y: np.array of target values
        X: observations
        seed: seed for reproducibility
    Returns:
        resampled y and X
    """
    sm = SMOTE(random_state=seed)
    X_resampled, y_resampled = sm.fit_resample(X, y)

    return y_resampled, X_resampled


def indices_by_class(y: np.array, num_classes: int = 3) -> List[np.array]:
    """
    get class indicees depending on given y
    Args:
        y: np.array of target values
        num_classes: number of classes in the classification task

    Returns:
        returns the indices divided following the different solubility classes
    """

    class_indices = []
    for class_ in range(num_classes):
        class_idx = np.where(y == class_)[0]
        class_indices.append(class_idx)
    return class_indices


def split_by_class(y: np.array, CV: int = 5, num_classes: int = 3) -> List[Tuple[List[np.array], List[np.array]]]:
    """

    Args:
        y: np.array of target values
        CV: number of cross-validation folds
        num_classes: number of classes in the classification task

    Returns:

    """
    splitted_classes_indices = indices_by_class(y, num_classes)

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


def create_subsample_train_csv(data_dir: str, features: np.array) -> None:
    """
    Creates a database based on the training data. Used for the GraphModel
    Args:
        data_dir: path to the data directory
        features: np.array of corresponding features

    Returns:

    """
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




