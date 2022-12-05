import os
from data_utils import *

if __name__ == "__main__":

    this_dir = os.getcwd()

    data_dir = os.path.join(this_dir, "data")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    # Initial dataset
    ids, smiles, targets = load_train_data(train_path)
    w_initial = sklearn.utils.class_weight.compute_class_weight('balanced', np.unique(targets), targets)
    print('Initial whole dataset: w =', w_initial)

    # Augmented dataset
    aug_id, aug_smiles, aug_targets = load_train_data(os.path.join(data_dir, 'augmented_ALLtrain.csv'))
    w_all_aug = sklearn.utils.class_weight.compute_class_weight('balanced', np.unique(targets), aug_targets)
    print('Augmented whole dataset: w =', w_all_aug)

    # SPLITS
    ids_train, smiles_train, targets_train = load_train_data(os.path.join(data_dir, 'split_train.csv'))
    ids_valid, smiles_valid, targets_valid = load_train_data(os.path.join(data_dir, 'split_valid.csv'))
    w_train = sklearn.utils.class_weight.compute_class_weight('balanced', np.unique(targets), targets_train)
    print('Split train set: w =', w_train)
    w_valid = sklearn.utils.class_weight.compute_class_weight('balanced', np.unique(targets), targets_valid)
    print('Split valid set: w =', w_valid)

    # AUGMENTED SPLITS
    aug_id_train, aug_smiles_train, aug_targets_train = load_train_data(os.path.join(data_dir, 'augmented_split_train.csv'))
    aug_id_valid, aug_smiles_valid, aug_targets_valid = load_train_data(os.path.join(data_dir, 'augmented_split_valid.csv'))
    w_train_aug = sklearn.utils.class_weight.compute_class_weight('balanced', np.unique(aug_targets_train), aug_targets_train)
    print('Augmented split train set: w =', w_train_aug)
    w_valid_aug = sklearn.utils.class_weight.compute_class_weight('balanced', np.unique(aug_targets_train), aug_targets_valid)
    print('Augmented split valid set: w =', w_valid_aug)

    # DOWNSAMPLED AUGMENTED SPLITS
    down_id_train, down_smiles_train, down_targets_train = load_train_data(
        os.path.join(data_dir, 'augmented_downsampled2_split_train.csv'))
    down_id_valid, down_smiles_valid, down_targets_valid = load_train_data(
        os.path.join(data_dir, 'augmented_downsampled2_split_valid.csv'))
    w_train_down = sklearn.utils.class_weight.compute_class_weight('balanced', np.unique(down_targets_train),
                                                                   down_targets_train)
    print('Augmented split train set: w =', w_train_down)
    w_valid_down = sklearn.utils.class_weight.compute_class_weight('balanced', np.unique(down_targets_valid),
                                                                   down_targets_valid)
    print('Augmented split valid set: w =', w_valid_down)
