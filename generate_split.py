from data_utils import *

if __name__ == "__main__":

    this_dir = os.getcwd()

    data_dir = os.path.join(this_dir, "data")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    # CREATION SPLIT DATASETS - new .csv files
    name_tr, name_val = create_split_csv(data_dir, "train.csv", downsampling_class2=False, p=0.6)
    # with downsampling of class 2
    name_tr_down, name_val_down = create_split_csv(data_dir, "train.csv", downsampling_class2=True, p=0.6)


    # AUGMENTATION OF EACH DATASET SEPARATELY - creation of new .csv files
    # train set
    ids_train, smiles_train, targets_train = load_train_data(os.path.join(data_dir, name_tr))
    _, aug_smiles_train, aug_targets_train = augment_smiles(ids_train, smiles_train, targets_train, data_dir, str('augmented_'+name_tr))
    # validation set
    ids_valid, smiles_valid, targets_valid = load_train_data(os.path.join(data_dir, name_val))
    _, aug_smiles_valid, aug_targets_valid = augment_smiles(ids_valid, smiles_valid, targets_valid, data_dir, str('augmented_'+name_val))

    # for the downsampled ones
    # downsampled train set
    ids_train_d, smiles_train_d, targets_train_d = load_train_data(os.path.join(data_dir, name_tr_down))
    _, down_smiles_train, down_targets_train = augment_smiles(ids_train_d, smiles_train_d, targets_train_d, data_dir, 'augmented_' + name_tr_down)
    # downsampled validation set
    ids_valid_d, smiles_valid_d, targets_valid_d = load_train_data(os.path.join(data_dir, name_val_down))
    _, down_smiles_valid, down_targets_valid = augment_smiles(ids_valid_d, smiles_valid_d, targets_valid_d, data_dir, 'augmented_' + name_val_down)

    # # CHECK THE NUMBER OF DATAPOINTS PER CLASS IN EACH SPLIT
    # ids_train, smiles_train, targets_train = load_train_data(os.path.join(data_dir, 'split_train.csv'))
    # ids_valid, smiles_valid, targets_valid = load_train_data(os.path.join(data_dir, 'split_valid.csv'))
    #
    # aug_smiles_train, aug_targets_train = load_train_data(os.path.join(data_dir, 'augmented_split_train.csv'))
    # aug_smiles_valid, aug_targets_valid = load_train_data(os.path.join(data_dir, 'augmented_split_valid.csv'))
    #
    # down_smiles_train, down_targets_train = load_train_data(os.path.join(data_dir, 'augmented_downsampled2_split_train.csv'))
    # down_smiles_valid, down_targets_valid = load_train_data(os.path.join(data_dir, 'augmented_downsampled2_split_valid.csv'))

    print('TRAIN SPLIT SET')
    print('Tot datapoints = ', targets_train.shape[0])
    print('Class 0 = ', sum(np.where(targets_train == 0, 1, 0)))
    print('Class 1 = ', sum(np.where(targets_train == 1, 1, 0)))
    print('Class 2 = ', sum(np.where(targets_train == 2, 1, 0)))

    print('VALIDATION SPLIT SET')
    print('Tot datapoints = ', targets_valid.shape[0])
    print('Class 0 = ', sum(np.where(targets_valid == 0, 1, 0)))
    print('Class 1 = ', sum(np.where(targets_valid == 1, 1, 0)))
    print('Class 2 = ', sum(np.where(targets_valid == 2, 1, 0)))

    print('************ AFTER AUGMENTATION ************')
    print('TRAIN SPLIT SET')
    print('Tot datapoints = ', aug_targets_train.shape[0])
    print('Class 0 = ', sum(np.where(aug_targets_train == 0, 1, 0)))
    print('Class 1 = ', sum(np.where(aug_targets_train == 1, 1, 0)))
    print('Class 2 = ', sum(np.where(aug_targets_train == 2, 1, 0)))

    print('VALIDATION SPLIT SET')
    print('Tot datapoints = ', aug_targets_valid.shape[0])
    print('Class 0 = ', sum(np.where(aug_targets_valid == 0, 1, 0)))
    print('Class 1 = ', sum(np.where(aug_targets_valid == 1, 1, 0)))
    print('Class 2 = ', sum(np.where(aug_targets_valid == 2, 1, 0)))

    print('************ AFTER DOWNSAMPLING + AUGMENTATION ************')
    print('TRAIN SPLIT SET')
    print('Tot datapoints = ', down_targets_train.shape[0])
    print('Class 0 = ', sum(np.where(down_targets_train == 0, 1, 0)))
    print('Class 1 = ', sum(np.where(down_targets_train == 1, 1, 0)))
    print('Class 2 = ', sum(np.where(down_targets_train == 2, 1, 0)))

    print('VALIDATION SPLIT SET')
    print('Tot datapoints = ', down_targets_valid.shape[0])
    print('Class 0 = ', sum(np.where(down_targets_valid == 0, 1, 0)))
    print('Class 1 = ', sum(np.where(down_targets_valid == 1, 1, 0)))
    print('Class 2 = ', sum(np.where(down_targets_valid == 2, 1, 0)))



    # ids, smiles, targets = load_train_data(train_path)
    #
    # qm_descriptors = smiles_to_qm_descriptors(smiles, data_dir)
    # (
    #     dataset,
    #     columns_info,
    # ) = nan_imputation(qm_descriptors, 0.0, standardization=False)
    #
    # make_umap(dataset, targets)

    # submission_ids, submission_smiles = load_test_data(test_path)

    # add new splitting like this:
    # split = split_by_class(targets)

    # DATA AUGMENTATION
