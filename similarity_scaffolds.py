import numpy as np
import pandas as pd
import os
import rdkit
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from rdkit import DataStructs
from rdkit import Chem
from data_utils import load_train_data, load_test_data


def get_scaffolds(smiles):
    """
    Calculates the scaffold of a list of molecules through the Murcko definition
    :param smiles: list of SMILES
    :return: list of respective scaffolds
    """
    scaffold_generator = rdkit.Chem.Scaffolds.MurckoScaffold

    scaffolds_list = []
    for smile in smiles:
        scaffold = scaffold_generator.MurckoScaffoldSmiles(smile)
        scaffolds_list.append(scaffold)

    return scaffolds_list


def diversity_scores(smiles1, smiles2, threshold=0.9, no_threshold=False):
    """
    Calculate the similarity between two lists of SMILES
    :param smiles1: list of SMILES
    :param smiles2: list of SMILES
    :param threshold: scalar value to decide if two SMILES are similar or not
    :param no_threshold: Bool
    :return: np.array of similarity scores
    """
    mols1 = []
    for smile in smiles1:
        mol = Chem.MolFromSmiles(smile)
        mols1.append(mol)
    mols2 = []
    for smile in smiles2:
        mol = Chem.MolFromSmiles(smile)
        mols2.append(mol)

    fps2 = [Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048) for mol in mols2]

    scores = np.array(
        list(map(lambda x: __compute_diversity(x, fps2) if x is not None else 0, mols1)))

    # Study the scores
    if not no_threshold:
        count = 0
        for score in scores:
            if score >= threshold:
                count += 1
        print('similar items: count=', count, '/', scores.shape, 'between tot1=', len(smiles1), 'tot2=', len(smiles2))

    return scores


def __compute_diversity(mol, fps):
    """
    Calculate the similarity score for a molecule and a list of fingerprints by using Tanimoto similarity
    :param mol: Mol object (a molecule)
    :param fps: List of fingerprints to compare the molecule to
    :return: mean of the similarity score between the molecule and all the fingerprints
    """
    ref_fps = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
    # comparing a single molecule to all the others in the second group
    dist = DataStructs.BulkTanimotoSimilarity(ref_fps, fps, returnDistance=True)
    # I get the average of the similarity with all the other molecules
    score = np.mean(dist)

    return score


def diversity_scores_singlepair(smiles1, smiles2, threshold=0.9, no_threshold=False):
    """
    Calculate the similarity between two SMILES
    :param smiles1: a SMILES
    :param smiles2: a SMILES
    :param threshold: scalar to determine if they are similar or not
    :param no_threshold: Bool
    :return: similarity score
    """
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    fp2 = [Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, 3, nBits=2048)]

    score = np.array(__compute_diversity_singlepair(mol1, fp2))

    return score


def __compute_diversity_singlepair(mol, fp):
    """
    Calculate the similarity score for a molecule and a fingerprint
    :param mol: Mol object
    :param fp: a single fingerprint
    :return: scalar Tanimoto similarity
    """
    ref_fp = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
    # comparing a single molecule to all the others in the second group
    dist = DataStructs.BulkTanimotoSimilarity(ref_fp, fp, returnDistance=True)

    return dist


class Scaffold:
    def __init__(self, smiles, ind, target):
        self.smiles = smiles
        self.num_mol = 1
        self.indices = list(ind)
        if target == 0:
            self.num_0 = 1
            self.num_1 = 0
            self.num_2 = 0
        elif target == 1:
            self.num_0 = 0
            self.num_1 = 1
            self.num_2 = 0
        elif target == 2:
            self.num_0 = 0
            self.num_1 = 0
            self.num_2 = 1

    def __len__(self):
        return self.num_mol

    def __add__(self, ind, target):
        self.num_mol += 1
        self.indices.append(ind)
        if target == 0:
            self.num_0 += 1
        elif target == 1:
            self.num_1 += 1
        elif target == 2:
            self.num_2 += 1

    def get_smile(self):
        return self.smiles

    def get_nmol(self):
        return self.num_mol

    def get_indices(self):
        return self.indices

    def get_class0(self):
        return self.num_0

    def get_class1(self):
        return self.num_1

    def get_class2(self):
        return self.num_2


def group_scaffolds(ids, scaffolds, targets, path):
    """
    Creation of a dictionary (and production of a .csv file) that contains all the unique sccaffolds of the train set,
    and the respective data
    :param ids: list of Ids of the molecules
    :param scaffolds: list of SMILES of the scaffolds
    :param targets: list of targets
    :param path: destination path where to save the .csv file
    :return: dictionary with all the unique scaffolds, the number of molecules with that same scaffold,
             their ids, the number of them that belong to class 0, 1 and 2
    """
    # Creating a dictionary with all the scaffolds
    dict_scaffolds = {}

    for (i, item, t) in zip(ids, scaffolds, targets):
        if item in dict_scaffolds:
            # identical scaffold
            dict_scaffolds[item].__add__(i, t)
        else:
            not_present = True
            for item_dict in dict_scaffolds:
                sim = diversity_scores_singlepair(item, item_dict)
                if sim > 0.95:
                    # very similar scaffold, we add it to the existent object
                    dict_scaffolds[item_dict].__add__(i, t)
                    not_present = False
                    break
            if not_present:
                # NEW SCAFFOLD - create a new object of the class Scaffold
                dict_scaffolds[item] = Scaffold(item, i, t)
                print('Added new scaffold')

    print('Created dictionary')
    # Create a .csv file with all the info
    # need to loop over the dictionary and retrieve all the data for every object of teìhe class
    data_scaffolds = []
    data_nmol = []
    data_ind = []
    data_0 = []
    data_1 = []
    data_2 = []
    for item in dict_scaffolds:
        data_scaffolds.append((dict_scaffolds[item].get_smile()))
        data_nmol.append((dict_scaffolds[item]).get_nmol())
        data_ind.append((dict_scaffolds[item]).get_indices())
        data_0.append((dict_scaffolds[item]).get_class0())
        data_1.append((dict_scaffolds[item]).get_class1())
        data_2.append((dict_scaffolds[item]).get_class2())
    data = {"Scaffold": data_scaffolds, "num_molecules": data_nmol, "num_class_0": data_0,
            "num_class_1": data_1, "num_class_2": data_2, "indices": data_ind}
    df = pd.DataFrame(data=data)
    df.to_csv(path, index=False)
    print('Created .csv file')

    print('Number of molecules =', len(scaffolds))
    print('Number of unique scaffolds =', len(dict_scaffolds))

    return dict_scaffolds


# Scaffolds test
class ScaffoldTest:
    def __init__(self, smiles, ind):
        self.smiles = smiles
        self.num_mol = 1
        self.indices = list(ind)

    def __len__(self):
        return self.num_mol

    def __add__(self, ind):
        self.num_mol += 1
        self.indices.append(ind)

    def get_smile(self):
        return self.smiles

    def get_nmol(self):
        return self.num_mol

    def get_indices(self):
        return self.indices


def group_scaffolds_test(ids, scaffolds, path):
    """
    Creation of a dictionary (and production of a .csv file) that contains all the unique sccaffolds of the TEST set,
    and the respective data
    :param ids: list of Ids of the molecules
    :param scaffolds: list of SMILES of the scaffolds
    :param path: destination path where to save the .csv file
    :return: dictionary with all the unique scaffolds, the number of molecules with that same scaffold,
             their ids
    """
    # Creating a dictionary with all the scaffolds
    dict_scaffolds = {}

    for (i, item) in zip(ids, scaffolds):
        if item in dict_scaffolds:
            # identical scaffold
            dict_scaffolds[item].__add__(i)
        else:
            not_present = True
            for item_dict in dict_scaffolds:
                sim = diversity_scores_singlepair(item, item_dict)
                if sim > 0.95:
                    # very similar scaffold, we add it to the existent object
                    dict_scaffolds[item_dict].__add__(i)
                    not_present = False
                    break
            if not_present:
                # NEW SCAFFOLD - create a new object of the class Scaffold
                dict_scaffolds[item] = ScaffoldTest(item, i)
                print('Added new scaffold')

    print('Created dictionary')
    # Create a .csv file with all the info
    # need to loop over the dictionary and retrieve all the data for every object of teìhe class
    data_scaffolds = []
    data_nmol = []
    data_ind = []

    for item in dict_scaffolds:
        data_scaffolds.append((dict_scaffolds[item].get_smile()))
        data_nmol.append((dict_scaffolds[item]).get_nmol())
        data_ind.append((dict_scaffolds[item]).get_indices())

    data = {"Scaffold": data_scaffolds, "num_molecules": data_nmol, "indices": data_ind}
    df = pd.DataFrame(data=data)
    df.to_csv(path, index=False)
    print('Created .csv file')

    print('Number of molecules =', len(scaffolds))
    print('Number of unique scaffolds =', len(dict_scaffolds))

    return dict_scaffolds


def comparison_scaffolds(dict_scaffolds, dict_scaffolds_test):
    """
    Check if there are the same scaffolds in train and test sets
    :param dict_scaffolds: dictionary with unique scaffolds in the train set
    :param dict_scaffolds_test: dictionary with unique scaffolds in the test set
    :return: number of similar scaffolds
    """
    count = 0
    for item_test in dict_scaffolds_test:
        for item_train in dict_scaffolds:
            if item_train == item_test:
                count += 1
            else:
                sim = diversity_scores_singlepair(item_train, item_test)
                if sim > 0.95:
                    # very similar scaffolds
                    count += 1
                    break

    print('Number of scaffolds in both train and test sets =', count)
    return count


if __name__ == "__main__":

    this_dir = os.getcwd()

    data_dir = os.path.join(this_dir, "data")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    seed = 13
    np.random.seed(seed)

    # SCAFFOLDS ANALYSIS

    print('TRAIN SET')
    ids, smiles, targets = load_train_data(train_path)
    scaffolds = get_scaffolds(smiles)
    print('Calculated scaffolds')
    path = os.path.join(data_dir, 'grouped_scaffolds.csv')
    dict_train = group_scaffolds(ids, scaffolds, targets, path)

    print('TEST SET')
    ids_test, smiles_test = load_test_data(test_path)
    scaffolds_test = get_scaffolds(smiles_test)
    print('Calculated scaffolds')
    path_test = os.path.join(data_dir, 'grouped_scaffolds_test.csv')
    dict_test = group_scaffolds_test(ids_test, smiles_test)

    # COMPARISON TRAIN AND TEST
    count = comparison_scaffolds(dict_train, dict_test)





