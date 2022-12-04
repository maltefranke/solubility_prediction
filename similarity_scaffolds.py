import numpy as np
import pandas as pd
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from data_utils import *
from rdkit import DataStructs


def get_scaffolds(smiles):
    scaffold_generator = rdkit.Chem.Scaffolds.MurckoScaffold

    scaffolds_list = []
    for smile in smiles:
        scaffold = scaffold_generator.MurckoScaffoldSmiles(smile)
        scaffolds_list.append(scaffold)

    return scaffolds_list


def diversity_scores(smiles1, smiles2, threshold=0.9, no_threshold=False):
    mols1 = []
    for smile in smiles1:
        mol = Chem.MolFromSmiles(smile)
        mols1.append(mol)
    mols2 = []
    for smile in smiles2:
        mol = Chem.MolFromSmiles(smile)
        mols2.append(mol)

    # fps1 = [Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048) for mol in mols1]
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
    ref_fps = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
    # comparing a single molecule to all the others in the second group
    dist = DataStructs.BulkTanimotoSimilarity(ref_fps, fps, returnDistance=True)
    # I get as score the average of the similarity with all the other molecules
    score = np.mean(dist)
    #dist_a = np.array(dist)
    #print(np.max(dist_a[dist_a < 1]))

    return score

#########################################################################
def diversity_scores_singlepair(smiles1, smiles2, threshold=0.9, no_threshold=False):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    fp2 = [Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, 3, nBits=2048)]

    score = np.array(__compute_diversity_singlepair(mol1, fp2))

    return score


def __compute_diversity_singlepair(mol, fp):
    ref_fp = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
    # comparing a single molecule to all the others in the second group
    dist = DataStructs.BulkTanimotoSimilarity(ref_fp, fp, returnDistance=True)

    return dist
#########################################################################


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

    # def __get_item__(self):
    #     obj_list = list(self.smiles, self.num_mol, self.indices, self.num_0, self.num_1, self.num_2)
    #     return obj_list

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

    # def __get_item__(self):
    #     obj_list = list(self.smiles, self.num_mol, self.indices)
    #     return obj_list

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
    # Creating a dictionary with all the scaffolds
    dict_scaffolds = {}

    for (i, item, t) in zip(ids, scaffolds, targets):
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


# Comparison scaffolds train and test sets
def comparison_scaffolds(dict_scaffolds, dict_scaffolds_test):
    # check if there are the same scaffolds in train and test sets
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
    path = data_dir + 'grouped_scaffolds.csv'
    dict_train = group_scaffolds(ids, scaffolds, targets, path)

    print('TEST SET')
    ids_test, smiles_test = load_test_data(test_path)
    scaffolds_test = get_scaffolds(smiles_test)
    print('Calculated scaffolds')
    path_test = data_dir + 'grouped_scaffolds_test.csv'
    dict_test = group_scaffolds_test(ids_test, smiles_test)

    # comparison train and test
    count = comparison_scaffolds(dict_train, dict_test)

    # SIMILARITY
    # we work only with smiles, not with MorganFP in the main
    # ids, smiles, targets = load_train_data(train_path)

    # print('******** TRAIN SET ANALYSIS ********')
    # # ********* SIMILARITY TRAIN SET **********
    # print('Similarity of the whole train set')
    # scores = diversity_scores(smiles, smiles)
    #
    # # ********* SCAFFOLDS STUDY **********
    # # Get the scaffolds of the whole train set
    # scaffolds = get_scaffolds(smiles)
    # print('Similarity of the scaffolds of the whole TRAIN SET')
    # scores_scaffolds = diversity_scores(scaffolds, scaffolds)
    #
    # # STUDY HOW THE SCAFFOLDS OF THE DIFFERENT CLASSES ARE
    # # look for some similarity
    # # divide the train set in classes
    # smiles2 = np.array(smiles)[np.where(targets == 2)[0]]
    # scaffolds2 = get_scaffolds(smiles2)
    # smiles1 = np.array(smiles)[np.where(targets == 1)[0]]
    # scaffolds1 = get_scaffolds(smiles1)
    # smiles0 = np.array(smiles)[np.where(targets == 0)[0]]
    # scaffolds0 = get_scaffolds(smiles0)
    #
    # # There are some identical scaffolds??
    #
    # # compute similarity of scaffolds (smiles) in each class
    # print('Similarity of scaffolds of the same class (train set)')
    # scores0 = diversity_scores(scaffolds0, scaffolds0)
    # scores1 = diversity_scores(scaffolds1, scaffolds1)
    # scores2 = diversity_scores(scaffolds2, scaffolds2)
    #
    # # REPEAT THE SAME PROCESS FOR THE TEST SET
    # print('******** TEST SET ANALYSIS ********')
    # ids, smiles_test = load_test_data(test_path)
    #
    # print('Similarity of the whole test set')
    # scores_test = diversity_scores(smiles_test, smiles_test)
    #
    # # ******** SCAFFOLD TEST STUDY ********
    # scaffolds_test = get_scaffolds(smiles_test)
    # print('Similarity of the scaffolds of the whole TEST SET')
    # scores_test = diversity_scores(scaffolds_test, scaffolds_test)
    #
    # # Now I can try to understand if by looking at the scaffolds of the test set
    # # I can find some similarities with the scaffolds of each class
    # # COMPARISON OF THE TEST SCAFFOLDS WITH EACH CLASS
    # # I need a different version of the similarity functions
    # print('Comparison of the TEST SET with the classes of the train set')
    #
    # scores_test0 = diversity_scores(scaffolds_test, scaffolds0)
    # scores_test1 = diversity_scores(scaffolds_test, scaffolds1)
    # scores_test2 = diversity_scores(scaffolds_test, scaffolds2)





