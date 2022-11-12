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

def up_down_sampling(y, X):

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

# TODO test if this function works
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
        molecule = Chem.MolFromSmiles(molecule)
        molecule = Chem.AddHs(molecule)
        AllChem.EmbedMolecule(molecule)
        AllChem.UFFOptimizeMolecule(molecule)
        molecule.GetConformer()

        atomic_nums = []
        atom_positions = []
        for i, atom in enumerate(molecule.GetAtoms()):
            atomic_nums.append(np.array(atom.GetAtomicNum()))

            positions = molecule.GetConformer().GetAtomPosition(i)
            atom_positions.append(np.array([positions.x, positions.y, positions.z]))

        all_atomic_nums.append(atomic_nums)
        all_positions.append(atom_positions)

    return all_atomic_nums, all_positions


# TODO maybe not even needed depending on the framework we use
def smiles_to_graph(smiles: List[str]):
    pass


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


if __name__ == "__main__":

    this_dir = os.getcwd()

    data_dir = os.path.join(this_dir, "data")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    # get data and transform smiles -> morgan fingerprint
    ids, smiles, targets = load_train_data(train_path)
    all_fps = smiles_to_morgan_fp(smiles)

    # see how balanced the data is and assign weights
    train_data_size = targets.shape[0]

    num_low = np.count_nonzero(targets == 0)
    num_medium = np.count_nonzero(targets == 1)
    num_high = np.count_nonzero(targets == 2)

    weights = [1 - num_low/train_data_size, 1 - num_medium/train_data_size, 1 - num_high/train_data_size]
    print(weights)

    seed = 13
    np.random.seed(seed)

    # we permutate/shuffle our data first
    p = np.random.permutation(targets.shape[0])
    all_fps = all_fps[p]
    targets = targets[p]

    # train an ensemble of ANNs, or load trained models if training was done previously
    model_checkpoints = ann_learning(all_fps, targets, ann_save_path=os.path.join(this_dir, "TestResults"),
                                     label_weights=weights)

    submission_ids, submission_smiles = load_test_data(test_path)
    X = smiles_to_morgan_fp(submission_smiles)
    input_dim = X.shape[-1]
    final_predictions = predict_ann_ensemble(X, input_dim, model_checkpoints)

    submission_file = os.path.join(this_dir, "predictions.csv")
    create_submission_file(submission_ids, final_predictions, submission_file)

