import os

import numpy as np

from typing import Tuple, List

import rdkit
from rdkit import Chem as Chem
from rdkit.Chem import AllChem as AllChem
from rdkit.Chem import rdFingerprintGenerator

from mordred import Calculator, descriptors

import h5py


def smiles_to_morgan_fp(smiles: List[str]) -> np.array:
    """
    Creation of morgan_fingerprints starting from the smiles of the molecules
    :param smiles:
    :return: array with Morgan fingerprints
    """
    fp_generator = rdFingerprintGenerator.GetMorganGenerator()
    all_fps = []
    for molecule in smiles:
        molecule = Chem.MolFromSmiles(molecule)
        fp = fp_generator.GetFingerprintAsNumPy(molecule)

        all_fps.append(fp)
    all_fps = np.array(all_fps)

    return all_fps


def smiles_to_qm_descriptors(
    smiles: List[str], data_dir: str, type_="train"
) -> np.array:
    """
    Creation or loading of the dataset containing features which denote physical/chemical quantities
    of the molecules
    :param smiles:
    :param data_dir:
    :param type_:
    :return:
    """
    # paths to the datasets
    if type_ == "train":
        qm_descriptor_file = os.path.join(
            data_dir, "train_qm_descriptors_3D.h5"
        )
    elif type_ == "test":
        qm_descriptor_file = os.path.join(
            data_dir, "test_qm_descriptors_3D.h5"
        )
    else:
        qm_descriptor_file = os.path.join(
            data_dir, "descriptors_collection.h5"
        )

    # creation of the dataset containing the descriptors
    if not os.path.exists(qm_descriptor_file):
        calc = Calculator(descriptors)

        mols = [Chem.MolFromSmiles(s) for s in smiles]

        df = calc.pandas(mols)

        qm_descriptors = df.to_numpy()

        qm_descriptors = qm_descriptors.astype(np.float64)
        with h5py.File(qm_descriptor_file, "w") as hf:
            hf.create_dataset("descriptors", data=qm_descriptors)
    # loading of the dataset
    else:
        with h5py.File(qm_descriptor_file, "r") as hf:
            qm_descriptors = hf["descriptors"][:]

    return qm_descriptors


def smiles_to_3d(smiles: List[str]) -> Tuple[List[np.array], List[np.array]]:
    """
    Transform a list of SMILES into their 3D representation using rdkit functions
    Args:
        :param smiles: list of SMILES string

    Returns:
        :return : tuple containing a list of atomic numbers and a list of corresponding atom positions
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
