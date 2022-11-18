import os
import csv
from typing import Tuple, List
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem as Chem
from rdkit.Chem import AllChem as AllChem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from sklearn.utils import resample
# from imblearn.over_sampling import SMOTE
from mordred import Calculator, descriptors

from models.ANN import *


def load_train_data(train_path: str) -> Tuple[List[str], List[str], np.array]:

    df = pd.read_csv(train_path)
    #df.pivot_table(index='sol_category', aggfunc='size').plot(kind='bar')

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
    X0_up = resample(X0, replace=True, n_samples=int(X.shape[0] * 0.25), random_state=13)
    X1_up = resample(X1, replace=True, n_samples=int(X.shape[0] * 0.25), random_state=13)

    # down-sample majority class
    X2_down = resample(X2, replace=False, n_samples=int(X.shape[0] * 0.5), random_state=13)

    X_balanced = np.concatenate((X0_up, X1_up, X2_down))
    y_balanced = np.concatenate((np.zeros(X0_up.shape[0], dtype=np.int8), np.ones(X1_up.shape[0], dtype=np.int8), 2*np.ones(X2_down.shape[0], dtype=np.int8)))
    print(X_balanced.shape, y_balanced.shape)

    return y_balanced, X_balanced

# def smote_algorithm(y, X):
    """
    Up-sample the minority class
    Args:
        y: classes values
        X: observations

    Returns:
        y_resampled
        X_resampled
    """
    # X_resampled, y_resampled = SMOTE().fit_resample(X, y)

    # return y_resampled, X_resampled


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
                atom_positions.append(np.array([positions.x, positions.y, positions.z]).tolist())

            all_atomic_nums.append(atomic_nums)
            all_positions.append(atom_positions)

        except:
            print("Molecule not processable")
            continue

    return all_atomic_nums, all_positions


def create_submission_file(ids: List[str], y_pred: np.array, path: str) -> None:
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


def calculate_class_weights(targets: np.array, num_classes: int = 3) -> List[float]:

    # see how balanced the data is and assign weights
    train_data_size = targets.shape[0]

    weights = [1 - np.count_nonzero(targets == int(i)) / train_data_size for i in range(num_classes)]

    return weights


if __name__ == "__main__":

    this_dir = os.getcwd()

    data_dir = os.path.join(this_dir, "data")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    # get data and transform smiles -> morgan fingerprint
    ids, smiles, targets = load_train_data(train_path)
    all_fps = smiles_to_morgan_fp(smiles)

    qm_descriptors = smiles_to_qm_descriptors(smiles, data_dir)

    seed = 13
    np.random.seed(seed)

    # we permutate/shuffle our data first
    p = np.random.permutation(targets.shape[0])
    all_fps = all_fps[p]
    targets = targets[p]



