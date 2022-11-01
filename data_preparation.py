import os
import numpy as np
from rdkit import Chem, AllChem


def load_data(path: str):
    pass


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
