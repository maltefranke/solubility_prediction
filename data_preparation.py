import os
import numpy as np
import pandas as pd
# from rdkit import Chem, AllChem


def load_train_data(train_path: str) -> tuple[list[str], list[str], list[int]]:

    df = pd.read_csv(train_path)

    ids = df["Id"].values.tolist()
    smiles = df["smiles"].values.tolist()
    targets = df["sol_category"].values.tolist()

    return ids, smiles, targets


def load_test_data(test_path: str) -> tuple[list[str], list[str]]:

    df = pd.read_csv(test_path)

    ids = df["Id"].values.tolist()
    smiles = df["smiles"].values.tolist()

    return ids, smiles


# TODO test if this function works
def smiles_to_3d(smiles: list[str]) -> tuple[list[np.array], list[np.array]]:
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


def smiles_to_graph(smiles: list[str]):
    pass


def create_submission_file(ids: list[int], y_pred: np.array, path: str) -> None:
    """
    Function to create the final submission file
    Args:
        ids: list of ids corresponding to the original data
        y_pred: np.array of model predictions
        path: path to where the csv should be saved

    Returns:
        None
    """
    pass


if __name__ == "__main__":
    this_dir = os.getcwd()

    data_dir = os.path.join(this_dir, "data")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    ids, smiles, targets = load_train_data(train_path)
    ids, smiles = load_test_data(test_path)

