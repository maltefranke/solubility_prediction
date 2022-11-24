import numpy as np
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from data_preparation import *
from rdkit import DataStructs


def get_scaffolds(smiles: List[str]) -> np.array:
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
    if no_threshold == False:
        count = 0
        for i, score in enumerate(scores):
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

if __name__ == "__main__":

    this_dir = os.getcwd()

    data_dir = os.path.join(this_dir, "data")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    seed = 13
    np.random.seed(seed)

    # TRAIN SET ANALYSIS
    # we work only with smiles, not with MorganFP in the main
    ids, smiles, targets = load_train_data(train_path)

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
    ids, smiles_test = load_test_data(test_path)
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

    # SCAFFOLDS ANALYSIS
    # Get the scaffolds of the whole train set
    print('TRAIN SET')
    scaffolds = get_scaffolds(smiles)
    # Creating a dictionary with all the scaffolds
    dict_scaffolds = {}
    for i, item in enumerate(scaffolds):
        if item in dict_scaffolds:
            dict_scaffolds[item] = dict_scaffolds[item]+1
        else:
            dict_scaffolds[item] = 1

    print('Number of molecules =', len(scaffolds))
    print('Number of unique scaffolds =', len(dict_scaffolds))

    # I do the same for the test set
    print('TEST SET')
    scaffolds_test = get_scaffolds(smiles_test)
    # Creating a dictionary with all the scaffolds
    dict_scaffolds_test = {}
    for i, item in enumerate(scaffolds_test):
        if item in dict_scaffolds_test:
            dict_scaffolds_test[item] = dict_scaffolds_test[item] + 1
        else:
            dict_scaffolds_test[item] = 1


    print('Number of molecules =', len(scaffolds_test))
    print('Number of unique scaffolds =', len(dict_scaffolds_test))

    # check the similarity of the unique scaffolds in the train test
    score = diversity_scores(dict_scaffolds.keys(), dict_scaffolds.keys(), no_threshold=True)

    # check if there are the same scaffolds in train and test sets
    # count = 0
    # for item_test in dict_scaffolds_test:
    #     for item_train in dict_scaffolds:
    #         check_similarity = False
    #         if item_train == item_test:
    #             count += 1
    #         else:
    #             check_similarity = True
    #     # comparing a scaffold in the test to all the ones in the train set
    #     scores = diversity_scores(item_test, dict_scaffolds.keys(), no_threshold=True)
    #     print(scores)
    #
    # print('Number of scaffolds in both train and test sets =', count)