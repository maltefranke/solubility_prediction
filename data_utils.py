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


def load_train_data(train_path: str):
    """
    Load the data from the .csv file with the train dataset
    :param train_path:
    :return:
    """

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
    """
    Load the data from the .csv file with the test dataset
    :param test_path:
    :return:
    """

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
    """

    :param targets:
    :param CV:
    :param num_classes:
    :return:
    """
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
    """

    :param data_dir:
    :param features:
    :return:
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


if __name__ == "__main__":

    """
    this_dir = os.getcwd()

    data_dir = os.path.join(this_dir, "data")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    ids, smiles, targets = load_train_data(train_path)

    from augmentation_utils import nan_imputation

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

    make_umap(
        dataset_shuffled[int(math.modf(len(targets) * 0.6)[1]) : -1, :],
        targets_shuffled[int(math.modf(len(targets) * 0.6)[1]) : -1],
    )


    """
