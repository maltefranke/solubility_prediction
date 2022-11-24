import os
import csv
from typing import Tuple, List
import numpy as np
import pandas as pd
import rdkit
import h5py
from rdkit import Chem as Chem
from rdkit.Chem import AllChem as AllChem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from sklearn.utils import resample
from mordred import Calculator, descriptors
from imblearn.over_sampling import SMOTE

from models.ANN import *


def load_train_data(train_path: str) -> Tuple[List[str], List[str], np.array]:

    df = pd.read_csv(train_path)

    ids = df["Id"].values.tolist()
    smiles = df["smiles"].values.tolist()
    targets = df["sol_category"].values.tolist()
    targets = np.array(targets)

    return ids, smiles, targets


def load_test_data(test_path: str) -> Tuple[List[str], List[str]]:

    df = pd.read_csv(test_path)

    ids = df["Id"].values.tolist()
    smiles = df["smiles"].values.tolist()

    return ids, smiles


def smiles_to_morgan_fp(smiles: List[str]) -> np.array:
    fp_generator = rdFingerprintGenerator.GetMorganGenerator()

    all_fps = []
    for molecule in smiles:
        molecule = Chem.MolFromSmiles(molecule)
        fp = fp_generator.GetFingerprintAsNumPy(molecule)

        all_fps.append(fp)
    all_fps = np.array(all_fps)

    return all_fps


"""
def smiles_to_qm_descriptors(smiles: List[str], data_dir: str) -> np.array:
    qm_descriptor_file = os.path.join(data_dir, "train_qm_descriptors.npy")
    if not os.path.exists(qm_descriptor_file):
        calc = Calculator(descriptors, ignore_3D=True)

        mols = [Chem.MolFromSmiles(s) for s in smiles]

        df = calc.pandas(mols)

        qm_descriptors = df.to_numpy()
        np.save(qm_descriptor_file, qm_descriptors)

    else:
        qm_descriptors = np.load(qm_descriptor_file, allow_pickle=True)

    return qm_descriptors
"""


def smiles_to_qm_descriptors(
    smiles: List[str], data_dir: str, type_="train"
) -> np.array:

    if type_ == "train":
        qm_descriptor_file = os.path.join(data_dir, "train_qm_descriptors.h5")
    elif type_ == "test":
        qm_descriptor_file = os.path.join(data_dir, "test_qm_descriptors.h5")
    else:
        qm_descriptor_file = os.path.join(
            data_dir, "descriptors_collection.h5"
        )

    if not os.path.exists(qm_descriptor_file):
        calc = Calculator(descriptors, ignore_3D=True)

        mols = [Chem.MolFromSmiles(s) for s in smiles]

        df = calc.pandas(mols)

        qm_descriptors = df.to_numpy()

        qm_descriptors = qm_descriptors.astype(np.float64)
        with h5py.File(qm_descriptor_file, "w") as hf:
            hf.create_dataset("descriptors", data=qm_descriptors)

    else:
        with h5py.File(qm_descriptor_file, "r") as hf:
            qm_descriptors = hf["descriptors"][:]
        # qm_descriptors = np.load(qm_descriptor_file, allow_pickle=True)

    return qm_descriptors


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


def smote_algorithm(y, X):
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


def smiles_to_3d(smiles: List[str]) -> Tuple[List[np.array], List[np.array]]:
    """
    Transform a list of SMILES into their 3D representation using rdkit functions
    Args:
        smiles: list of SMILES string

    Returns:
        tuple containing a list of atomic numbers and a list of corresponding atom positions
    """
    # following https://stackoverflow.com/questions/71915443/rdkit-coordinates-for-atoms-in-a-molecule
    all_atomic_nums = []
    all_positions = []
    for molecule in smiles:
        try:
            molecule = Chem.MolFromSmiles(molecule)
            molecule = Chem.AddHs(molecule)
            AllChem.EmbedMolecule(molecule)
            AllChem.UFFOptimizeMolecule(molecule)
            molecule.GetConformer()

            atomic_nums = []
            atom_positions = []
            for i, atom in enumerate(molecule.GetAtoms()):
                # atomic_nums.append(np.array(atom.GetAtomicNum()).item())
                atomic_nums.append(int(atom.GetAtomicNum()))

                positions = molecule.GetConformer().GetAtomPosition(i)
                atom_positions.append(
                    np.array([positions.x, positions.y, positions.z]).tolist()
                )

            all_atomic_nums.append(atomic_nums)
            all_positions.append(atom_positions)

        except:
            print("Molecule not processable")
            continue

    return all_atomic_nums, all_positions


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


def calculate_class_weights(
    targets: np.array, num_classes: int = 3
) -> List[float]:

    # see how balanced the data is and assign weights
    train_data_size = targets.shape[0]

    weights = [
        1 - np.count_nonzero(targets == int(i)) / train_data_size
        for i in range(num_classes)
    ]

    return weights


def create_subsample_train_csv(data_dir: str):
    path = os.path.join(data_dir, "random_undersampling.csv")
    train_path = os.path.join(data_dir, "train.csv")
    ids, smiles, targets = load_train_data(train_path)

    p = np.random.permutation(targets.shape[0])
    smiles = np.array(smiles)[p]
    targets = targets[p]

    # dividing the data in subsets depending on the class
    smiles_per_class = []
    for class_idx in range(3):
        smiles_class_i = np.array(smiles)[
            np.where(targets == class_idx)
        ].tolist()
        smiles_per_class.append(smiles_class_i)

    min_len = min([len(i) for i in smiles_per_class])

    with open(path, "w") as subsampling_file:
        writer = csv.writer(subsampling_file, delimiter=",")
        writer.writerow(["smiles", "sol_category"])

        for idx, smiles_class_i in enumerate(smiles_per_class):
            subsampled_smiles = smiles_class_i[:min_len]

            for temp_smiles in subsampled_smiles:
                writer.writerow([temp_smiles, idx])


# def nan_elimination(data):
#     """
#
#     Parameters
#     ----------
#     data : dataset to be checked
#
#     Returns
#     -------
#     data : dataset without nan values
#
#     """
#
#     N, M = data.shape
#     columns = []
#     list_of_counts = []
#     modified_columns = []
#     missing = np.zeros(M)
#     for i in range(M):
#         missing[i] = len(np.where(np.isnan(data[:, i]))[0]) / N
#
#         if missing[i] > 0.0:
#             columns.append(i)
#
#     data = np.delete(data, columns, axis=1)
#
#     return data, columns


def nan_imputation(data, nan_tolerance=0.5):
    """
    Function that removes columns with too many nan values (depending on the tolerance) and standardizes
    the data substituting the median to the nan values.
    It doesn't touch the categorical features.
    :param nan_tolerance: percentage of nan we want to accept for each column
    :param data: list with only qm_descriptors!!!!!!!!
    :return:
    """

    N, M = data.shape

    modified_columns = []
    # list that contains 0 if the col is removed, 1 if it is categorical, # 2 if it needs to be standardized

    standardization_data_train = np.empty((M, 3))
    # matrix that contains median, mean and std for each column that has been standardized

    erased = []
    # list of erased column (to substitute the necessary reduction of i and M in the for loop)

    for i in range(M):
        nan_percentage = len(np.where(np.isnan(data[:, i]))[0]) / N

        if nan_percentage > nan_tolerance:  # remove column
            # data = np.delete(data, i, axis=1)
            modified_columns.append(0)
            # i -= 1
            # M -= 1
            # check if this work, not sure!!
            erased.append(i)
        else:  # do not remove this column
            if check_categorical(
                data[:, i]
            ):  # if it is categorical, don't do anything
                modified_columns.append(1)

            else:  # it needs to be standardized
                modified_columns.append(2)
                median = np.nanmedian(data[:, i])
                # standardization (shouldn't affect nan values)
                data[:, 1], mean, std = standardize(data[:, 1])
                # replace nan with median
                data[:, i] = np.where(np.isnan(data[:, i]), median, data[:, i])
                standardization_data_train[i, :] = median, mean, std

    data = np.delete(data, erased, axis=1)

    return data, np.array(modified_columns), standardization_data_train


def standardize(x):
    """
    Given a column x, it calculates mean and std ignoring nan values and applies standardization
    :param x:
    :return: standardized x, mean, std
    """
    mean, std = np.nanmean(x, axis=0), np.nanstd(x, axis=0)
    # x = (x - mean) / std
    x = x - mean
    if std > 0:
        x = x / std

    return x, mean, std


def standardize_qm_test(data, columns_info, standardization_data):

    N, M = data.shape

    for i in range(M):
        if columns_info[i] == 0:
            data = np.delete(data, i, axis=1)
            i -= 1
            M -= 1
        elif columns_info[i] == 2:
            median = standardization_data[i, 0]
            mean = standardization_data[i, 1]
            std = standardization_data[i, 2]
            data[i, :] = (data[i, :] - mean) / std
            data[:, i] = np.where(np.isnan(data[:, i]), median, data[:, i])

    return data


def check_categorical(column):
    """
    Function that checks if a columns contains categorical feature or not (ignoring the nan values)
    :param column:
    :return: Bool
    """
    # removing nan values and substituting them with 0
    column_removed = np.where(np.isnan(column), 0, column)
    # calculating the modulus of the column
    modulus = np.mod(np.abs(column_removed), 1.0)

    if all(item == 0 for item in modulus):
        return True

    return False


# def find_categorical(data, modified=np.array([])):
#     data_copy = data
#
#     # find nans
#     idxs = np.where(np.isnan(data))
#     data_copy[idxs[0], idxs[1]] = 0.0
#
#     data_copy = np.abs(data_copy)
#     rem = np.mod(data_copy, 1.0)
#
#     idx = np.argwhere(np.all(rem[..., :] == 0, axis=0))
#     # idx = np.concatenate(idx, modified)
#
#     # idx = idx.unique()
#
#     return idx


if __name__ == "__main__":

    this_dir = os.getcwd()

    data_dir = os.path.join(this_dir, "data")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    # get data and transform smiles -> morgan fingerprint
    ids, smiles = load_test_data(test_path)
    all_fps = smiles_to_morgan_fp(smiles)

    qm_descriptors = smiles_to_qm_descriptors(smiles, data_dir)

    seed = 13
    np.random.seed(seed)

    # we permutate/shuffle our data first
    # p = np.random.permutation(targets.shape[0])
    # all_fps = all_fps[p]
    # targets = targets[p]
