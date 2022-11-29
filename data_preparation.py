import os
import csv
import numpy as np
import pandas as pd
import rdkit
import h5py
import random
import matplotlib.pyplot as plt

from typing import Tuple, List

from rdkit import Chem as Chem
from rdkit.Chem import AllChem as AllChem
from rdkit.Chem import rdFingerprintGenerator

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.utils import resample

from mordred import Calculator, descriptors

from imblearn.over_sampling import SMOTE

from utils import *


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
    """
    Creation of morgan_fingerprints starting from the smiles of the molecules
    """
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
    Creation or loading of the dataset containing features which denote physical/chemical quantities of the molecules
    """

    # paths to the datasets
    if type_ == "train":
        qm_descriptor_file = os.path.join(data_dir, "train_qm_descriptors.h5")
    elif type_ == "test":
        qm_descriptor_file = os.path.join(data_dir, "test_qm_descriptors.h5")
    else:
        qm_descriptor_file = os.path.join(
            data_dir, "descriptors_collection.h5"
        )

    # creation of the dataset containing the descriptors
    if not os.path.exists(qm_descriptor_file):
        calc = Calculator(descriptors, ignore_3D=True)

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


def preprocessing(ids, smiles, data_dir, degree=1, fps=False):
    """
    Sequence of functions to transform the dataset
    """

    # introduce descriptors
    qm_descriptors = smiles_to_qm_descriptors(smiles, data_dir)

    # we perform standardization only on qm descriptors!
    dataset, columns_info = nan_imputation(qm_descriptors, 0.0, cat_del=True)

    # if degree > 1:
    #    dataset = build_poly(dataset, columns_info, degree) # included in transformations

    if fps == True:
        # add morgan fingerprints
        all_fps = smiles_to_morgan_fp(smiles)
        dataset = np.concatenate((dataset, all_fps), axis=1)

    return dataset, columns_info


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


def build_poly(x, columns_info, degree, pairs=False):
    """Polynomial basis functions for input data x, for j=0 up to j=degree.
    Optionally can add square or cube roots of x as additional features,
    or the basis of products between the features.
    Args:
        x: numpy array of shape (N,), N is the number of samples
        degree: integer
        pairs: boolean
    Returns:
        poly: numpy array of shape (N,d+1)
    """
    # I have already removed nan columns
    # columns_info = np.delete(columns_info, np.where(columns_info == 0)[0]) # already done in transformation

    poly = np.ones((len(x), 1))
    for deg in range(1, degree + 1):
        if deg > 1:
            transformed = np.power(x[:, np.where(columns_info == 2)[0]], deg)
        else:
            # if deg==1, the standardization has already been made. Moreover, we should not loose categorical features
            transformed = x
        poly = np.c_[poly, transformed]
        new_cols = 2 * np.ones(tranformed.shape[1], dtype=int)
        columns_info = np.concatenate((columns_info, new_cols))

    if pairs:
        for i in range(x.shape[1]):
            for j in range(i + 1, x.shape[1]):
                if columns_info[i] == columns_info[j] == 2:
                    transformed = x[:, i] * x[:, j]
                    poly = np.c_[poly, transformed]
                    new_cols = 2 * np.ones(tranformed.shape[1], dtype=int)
                    columns_info = np.concatenate((columns_info, new_cols))

    return poly, columns_info


def transformation(
    data, columns_info, standardization=True, test=False, degree=1, pairs=False
):

    data = np.delete(data, np.where(columns_info == 0)[0], axis=1)
    # now eliminated both in test and training
    columns_info = np.delete(columns_info, np.where(columns_info == 0)[0])

    # correct nans in test

    if test == True:
        N, M = data.shape
        for i in range(M):
            if columns_info[i] == 2:
                median = np.nanmedian(data[:, i])
                data[:, i] = np.where(np.isnan(data[:, i]), median, data[:, i])

    # build poly
    if degree > 1 or pairs == True:
        data, columns_info = build_poly(data, columns_info, degree, pairs)

    if standardization == True:
        data[
            :, np.where(columns_info == 2)[0]
        ] = StandardScaler().fit_transform(
            data[:, np.where(columns_info == 2)[0]]
        )

    return data


def nan_imputation(
    data,
    nan_tolerance=0.5,
    standardization=True,
    cat_del=False,
    degree=1,
    pairs=False,
):
    """
    Function that removes columns with too many nan values (depending on the tolerance) and standardizes
    the data substituting the median to the nan values.
    It doesn't touch the categorical features.
    :param nan_tolerance: percentage of nan we want to accept for each column
    :param data: list with only qm_descriptors
    :return:
    """

    N, M = data.shape

    columns_info = []
    # list that contains 0 if the col is removed, 1 if it is categorical, # 2 if it needs to be standardized

    for i in range(M):
        nan_percentage = len(np.where(np.isnan(data[:, i]))[0]) / N

        if nan_percentage > nan_tolerance:  # remove column
            columns_info.append(0)

        else:
            if check_categorical(
                data[:, i]
            ):  # if it is categorical, don't do anything or delete
                if cat_del == True:
                    columns_info.append(0)
                else:
                    columns_info.append(1)

            else:  # it needs to be standardized
                columns_info.append(2)
                median = np.nanmedian(data[:, i])
                # replace nan with median
                data[:, i] = np.where(np.isnan(data[:, i]), median, data[:, i])
                # if standardization == True:
                # standardization
                # data[:, i], mean, std = standardize(data[:, i])
                # standardization_data_train[i, :] = median, mean, std

    columns_info = np.array(columns_info)
    data = transformation(data, columns_info, standardization, degree, pairs)
    return data, columns_info


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


def randomize_smiles(smiles, random_type="rotated", isomericSmiles=True):
    """
    From: https://github.com/undeadpixel/reinvent-randomized and https://github.com/GLambard/SMILES-X
    Returns a random SMILES given a SMILES of a molecule.
    :param mol: A Mol object
    :param random_type: The type (unrestricted, restricted, rotated) of randomization performed.
    :return : A random SMILES string of the same molecule or None if the molecule is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None

    if random_type == "unrestricted":
        return Chem.MolToSmiles(
            mol, canonical=False, doRandom=True, isomericSmiles=isomericSmiles
        )
    elif random_type == "restricted":
        new_atom_order = list(range(mol.GetNumAtoms()))
        random.shuffle(new_atom_order)
        random_mol = Chem.RenumberAtoms(mol, newOrder=new_atom_order)
        return Chem.MolToSmiles(
            random_mol, canonical=False, isomericSmiles=isomericSmiles
        )
    elif random_type == "rotated":
        n_atoms = mol.GetNumAtoms()
        rotation_index = random.randint(0, n_atoms - 1)
        atoms = list(range(n_atoms))
        new_atoms_order = (
            atoms[rotation_index % len(atoms) :]
            + atoms[: rotation_index % len(atoms)]
        )
        rotated_mol = Chem.RenumberAtoms(mol, new_atoms_order)
        return Chem.MolToSmiles(
            rotated_mol, canonical=False, isomericSmiles=isomericSmiles
        )
    raise ValueError("Type '{}' is not valid".format(random_type))


def augment_smiles(
    smiles: List[str], targets: np.array, data_dir: str
) -> Tuple[List[str], np.array]:
    augmentations_path = os.path.join(data_dir, "augmented_smiles.csv")
    if not os.path.exists(augmentations_path):
        class_indices = indices_by_class(targets)

        augmentations = []
        for iteration, class_idx in enumerate(class_indices):
            smiles_class_i = np.array(smiles)[class_idx].tolist()

            if iteration == 0 or iteration == 1:
                augmentations_by_class = [
                    list(set([randomize_smiles(i) for j in range(200)]))
                    for i in smiles_class_i
                ]
            else:
                augmentations_by_class = [
                    list(set([randomize_smiles(i) for j in range(5)]))
                    for i in smiles_class_i
                ]

            augmentations.append(augmentations_by_class)

        augmentations = [np.concatenate(i) for i in augmentations]
        augmentations_targets = [
            np.array([idx for i in augmentation])
            for idx, augmentation in enumerate(augmentations)
        ]

        augmentations = np.concatenate(augmentations).tolist()
        augmentations_targets = np.concatenate(augmentations_targets)

        data = {"smiles": augmentations, "sol_category": augmentations_targets}

        df = pd.DataFrame(data=data)
        df.to_csv(augmentations_path, index=False)

    final_df = pd.read_csv(augmentations_path)

    final_smiles = final_df["smiles"].tolist()
    final_targets = final_df["sol_category"].to_numpy()

    return final_smiles, final_targets


def PCA_application(dataset, targets):

    X = pd.DataFrame(dataset)
    y = pd.DataFrame(targets)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    std = StandardScaler()
    X_train_std = std.fit_transform(X_train)
    X_test_std = std.transform(X_test)

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

    make_umap(dataset, targets)
    # submission_ids, submission_smiles = load_test_data(test_path)

    # add new splitting like this:
    # split = split_by_class(targets)

    # DATA AUGMENTATION
