from rdkit import DataStructs

from data_preparation import *
import rdkit


# Convert the SMILES into Morgan fingerprint byte string.
def get_query_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES {smiles!r}")

    query_rd_fp = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useChirality=0,
                                                                      useBondTypes=1, useFeatures=0)
    return query_rd_fp


def __compute_diversity(smiles, fps):
    ref_fps = get_query_fp(smiles)
    dist = DataStructs.BulkTanimotoSimilarity(ref_fps, fps, returnDistance=True)
    score = np.mean(dist)

    return score

if __name__ == "__main__":

    this_dir = os.getcwd()

    data_dir = os.path.join(this_dir, "data")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    # get data and transform smiles -> morgan fingerprint
    ids_train, smiles_train, targets = load_train_data(train_path)
    train_fps = smiles_to_morgan_fp(smiles_train)

    ids_test, smiles_test = load_test_data(test_path)
    test_fps = smiles_to_morgan_fp(smiles_test)

    # Measure similarity of the train set
    threshold = 0.5
    query_fp = train_fps[1]
    target_fingerprints = train_fps
    target_ids = ids_train

    # Compute the score with each of the targets
    query_id = 1
    scores = __compute_diversity(smiles_train[1], target_fingerprints)
    print(scores)
    # Find the records with a high enough similarity.
    for i, score in enumerate(scores):
        if score >= threshold:
            # Need to get the corresponding id
            target_id = target_ids[i]
            has_match = True
            print(f"{query_id}\t{target_id}\t{score:.3f}")


    # qm_descriptors = smiles_to_qm_descriptors(smiles, data_dir)

    seed = 13
    np.random.seed(seed)

    # we permute/shuffle our data first
    # p = np.random.permutation(targets.shape[0])
    # all_fps = all_fps[p]
    # targets = targets[p]
