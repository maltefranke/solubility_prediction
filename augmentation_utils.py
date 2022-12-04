from data_utils import *
from conversion_smiles_utils import *

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
        new_cols = 2 * np.ones(transformed.shape[1], dtype=int)
        columns_info = np.concatenate((columns_info, new_cols))

    if pairs:
        for i in range(x.shape[1]):
            for j in range(i + 1, x.shape[1]):
                if columns_info[i] == columns_info[j] == 2:
                    transformed = x[:, i] * x[:, j]
                    poly = np.c_[poly, transformed]
                    new_cols = 2 * np.ones(transformed.shape[1], dtype=int)
                    columns_info = np.concatenate((columns_info, new_cols))

    return poly, columns_info


def logarithm(dataset, columns_info, log_trans=None):
    skewness = []
    to_transform = []

    if (
        log_trans == None
    ):  # so that I trasform the same columns in the test set
        for i in range(dataset.shape[1]):
            if (
                columns_info[i] == 2
                and len(np.where(np.sign(dataset[:, i]) == -1.0)[0]) == 0
            ):
                skewness.append(sc.stats.skew(dataset[:, i]))
                if np.abs(skewness[-1]) >= 1:
                    to_transform.append(i)
    else:
        to_transform = log_trans
    print(len(to_transform))
    dataset[:, to_transform] = np.log1p(dataset[:, to_transform])

    return dataset, to_transform


def transformation(
    data,
    columns_info,
    standardization=True,
    test=False,
    degree=1,
    pairs=False,
    log_trans=None,
    log=True,
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
    if log == True:
        data, log_trans = logarithm(data, columns_info, log_trans)

    # build poly
    if degree > 1 or pairs == True:
        data, columns_info = build_poly(data, columns_info, degree, pairs)

    if standardization == True:
        data[
            :, np.where(columns_info == 2)[0]
        ] = StandardScaler().fit_transform(
            data[:, np.where(columns_info == 2)[0]]
        )

    return data, log_trans


def nan_imputation(
    data,
    nan_tolerance=0.5,
    standardization=True,
    cat_del=False,
    degree=1,
    pairs=False,
    log=True,
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
    data, log_trans = transformation(
        data,
        columns_info,
        standardization=standardization,
        degree=degree,
        pairs=pairs,
        log=log,
    )
    return data, columns_info, log_trans


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


def augment_smiles(ids, smiles, targets, data_dir, name_file):
    """
    Addition of the rotations of a molecule depending on the class it belongs to.
    :param smiles: of the dataset we want to augment
    :param targets: of the dataset we want to augment
    :param data_dir: where we want to save the new file
    :param name_file: name of the file we want to save (augmented smiles)
    :return: augmented_smiles, augmented_targets
    """
    augmentations_path = os.path.join(data_dir, name_file)
    if not os.path.exists(augmentations_path):
        class_indices = indices_by_class(targets)

        augmentations = set()
        augmentations_id = []
        augmentations_targets = []
        for iteration, class_idx in enumerate(class_indices):
            smiles_class_i = np.array(smiles)[class_idx].tolist()
            ids_class_i = np.array(ids)[class_idx].tolist()
            targets_class_i = np.array(targets)[class_idx].tolist()

            if iteration == 0 or iteration == 1:
                len_smiles = len(augmentations)
                for (ind, i, t) in zip(ids_class_i, smiles_class_i, targets_class_i):
                    # Adding the original SMILES
                    augmentations.add(i)
                    len_smiles += 1
                    augmentations_id.append(ind)
                    augmentations_targets.append(t)
                    # Adding the rotations
                    for j in range(200):
                        augmentations.add(randomize_smiles(i))
                        if len(augmentations) == len_smiles + 1:
                            len_smiles += 1
                            augmentations_id.append(f"{ind}{j}")
                            augmentations_targets.append(t)

            else:
                len_smiles = len(augmentations)
                for (ind, i, t) in zip(ids_class_i, smiles_class_i, targets_class_i):
                    # Adding the original SMI.add(i)
                    len_smiles += 1
                    augmentations_id.append(ind)
                    augmentations_targets.append(t)
                    # Adding the rotations
                    for j in range(5):
                        augmentations.add(randomize_smiles(i))
                        if len(augmentations) == len_smiles + 1:
                            len_smiles += 1
                            augmentations_id.append(f"{ind}{j}")
                            augmentations_targets.append(t)

        augmentations = list(augmentations)

        data = {"Id": augmentations_id, "smiles": augmentations, "sol_category": augmentations_targets}

        # creation of the augmented dataset
        df = pd.DataFrame(data=data)
        df.to_csv(augmentations_path, index=False)

    final_df = pd.read_csv(augmentations_path)

    final_id = final_df["Id"].tolist()
    final_smiles = final_df["smiles"].tolist()
    final_targets = final_df["sol_category"].to_numpy()

    return final_id, final_smiles, final_targets


def create_split_csv(data_dir, file_name, downsampling_class2=False, p=0.6):
    """
    It creates 2 .csv files containing a random split of the given dataset into train (70%) and
    validation (30%) set.
    :param downsampling_class2:
    :param data_dir:
    :param file_name:
    :param p: percentage of datapoints in class 2 to keep
    :return:
    """
    data_path = os.path.join(data_dir, file_name)
    df = pd.read_csv(data_path)
    # random shuffle
    df = sklearn.utils.shuffle(df)
    # read ids, smiles, targets
    ids = df["Id"].values.tolist()
    smiles = df["smiles"].values.tolist()
    targets = df["sol_category"].values.tolist()
    targets = np.array(targets)

    if downsampling_class2:
        ind_2 = np.where(targets == 2)[0]
        ind_2_to_delete = ind_2[int(p * ind_2.shape[0]) : ind_2.shape[0]]
        df.drop(df.index[ind_2_to_delete], inplace=True)
        # re-acquiring data from the down-sampled dataset
        ids = df["Id"].values.tolist()
        smiles = df["smiles"].values.tolist()
        targets = df["sol_category"].values.tolist()
        targets = np.array(targets)

    # assign 70% - 30% of the data to train - validation
    # (it is random, it doesn't depend on the classes)
    N = targets.shape[0]
    ind_train = np.arange(int(N*0.7))
    ind_valid = np.arange(int(N*0.7), N)

    ids_train = np.array(ids)[ind_train]
    ids_valid = np.array(ids)[ind_valid]
    smiles_train = np.array(smiles)[ind_train]
    smiles_valid = np.array(smiles)[ind_valid]
    targets_train = targets[ind_train]
    targets_valid = targets[ind_valid]

    # creation csv files
    dataset_train = {"Id": ids_train, "smiles": smiles_train, "sol_category": targets_train}
    dataset_valid = {"Id": ids_valid, "smiles": smiles_valid, "sol_category": targets_valid}

    name_train_file = 'split_train.csv'
    name_valid_file = 'split_valid.csv'

    if downsampling_class2:
        name_train_file = 'downsampled2_' + name_train_file
        name_valid_file = 'downsampled2_' + name_valid_file
        # name_test_file = 'downsampled2_' + name_test_file

    dataset_train = {
        "Id": ids_train,
        "smiles": smiles_train,
        "sol_category": targets_train,
    }
    dataset_valid = {
        "Id": ids_valid,
        "smiles": smiles_valid,
        "sol_category": targets_valid,
    }

    name_train_file = "split_train.csv"
    name_valid_file = "split_valid.csv"

    if downsampling_class2:
        name_train_file = "downsampled2_" + name_train_file
        name_valid_file = "downsampled2_" + name_valid_file

    df_train = pd.DataFrame(data=dataset_train)
    df_train.to_csv(os.path.join(data_dir, name_train_file), index=False)
    df_valid = pd.DataFrame(data=dataset_valid)
    df_valid.to_csv(os.path.join(data_dir, name_valid_file), index=False)

    return name_train_file, name_valid_file


if __name__ == "__main__":

    this_dir = os.getcwd()

    data_dir = os.path.join(this_dir, "data")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    """
    # CREATION AUGMENTED DATASET WITH IDs
    ids, smiles, targets = load_train_data(train_path)
    aug_id, aug_smiles, aug_targets = augment_smiles(ids, smiles, targets, data_dir, 'augmented_ALLtrain.csv')

    # # CREATION SPLIT DATASETS - new .csv files
    # name_tr, name_val, name_te = create_split_csv(data_dir, "train.csv", downsampling_class2=True, p=0.6)
    #
    # # AUGMENTATION OF EACH DATASET SEPARATELY - creation of new .csv files
    # ids_train, smiles_train, targets_train = load_train_data(os.path.join(data_dir, name_tr))
    # aug_smiles_train, aug_targets_train = augment_smiles(smiles_train, targets_train, data_dir, 'augmented_'+name_tr)
    #
    # ids_valid, smiles_valid, targets_valid = load_train_data(os.path.join(data_dir, name_val))
    # aug_smiles_valid, aug_targets_valid = augment_smiles(smiles_valid, targets_valid, data_dir, 'augmented_'+name_val)
    #
    # ids_test, smiles_test, targets_test = load_train_data(os.path.join(data_dir, name_te))
    # aug_smiles_test, aug_targets_test = augment_smiles(smiles_test, targets_test, data_dir, 'augmented_' + name_te)

    # # CHECK THE NUMBER OF DATAPOINTS PER CLASS IN EACH SPLIT
    # ids_train, smiles_train, targets_train = load_train_data(os.path.join(data_dir, 'split_train.csv'))
    # ids_valid, smiles_valid, targets_valid = load_train_data(os.path.join(data_dir, 'split_valid.csv'))
    # ids_test, smiles_test, targets_test = load_train_data(os.path.join(data_dir, 'split_test.csv'))
    #
    # aug_smiles_train, aug_targets_train = load_train_data(os.path.join(data_dir, 'augmented_split_train.csv'))
    # aug_smiles_valid, aug_targets_valid = load_train_data(os.path.join(data_dir, 'augmented_split_valid.csv'))
    # aug_smiles_test, aug_targets_test = load_train_data(os.path.join(data_dir, 'augmented_split_test.csv'))
    #
    # down_smiles_train, down_targets_train = load_train_data(os.path.join(data_dir, 'augmented_downsampled2_split_train.csv'))
    # down_smiles_valid, down_targets_valid = load_train_data(os.path.join(data_dir, 'augmented_downsampled2_split_valid.csv'))
    # down_smiles_test, down_targets_test = load_train_data(os.path.join(data_dir, 'augmented_downsampled2_split_test.csv'))

    print('************ AUGMENTED ALL TRAIN SET ************')
    print('Tot datapoints = ', aug_targets.shape[0])
    print('Class 0 = ', sum(np.where(aug_targets == 0, 1, 0)))
    print('Class 1 = ', sum(np.where(aug_targets == 1, 1, 0)))
    print('Class 2 = ', sum(np.where(aug_targets == 2, 1, 0)))

    # print('TRAIN SPLIT SET')
    # print('Tot datapoints = ', targets_train.shape[0])
    # print('Class 0 = ', sum(np.where(targets_train == 0, 1, 0)))
    # print('Class 1 = ', sum(np.where(targets_train == 1, 1, 0)))
    # print('Class 2 = ', sum(np.where(targets_train == 2, 1, 0)))
    #
    # print('VALIDATION SPLIT SET')
    # print('Tot datapoints = ', targets_valid.shape[0])
    # print('Class 0 = ', sum(np.where(targets_valid == 0, 1, 0)))
    # print('Class 1 = ', sum(np.where(targets_valid == 1, 1, 0)))
    # print('Class 2 = ', sum(np.where(targets_valid == 2, 1, 0)))
    #
    # print('TEST SPLIT SET')
    # print('Tot datapoints = ', targets_test.shape[0])
    # print('Class 0 = ', sum(np.where(targets_test == 0, 1, 0)))
    # print('Class 1 = ', sum(np.where(targets_test == 1, 1, 0)))
    # print('Class 2 = ', sum(np.where(targets_test == 2, 1, 0)))
    #
    # print('************ AFTER AUGMENTATION ************')
    # print('TRAIN SPLIT SET')
    # print('Tot datapoints = ', aug_targets_train.shape[0])
    # print('Class 0 = ', sum(np.where(aug_targets_train == 0, 1, 0)))
    # print('Class 1 = ', sum(np.where(aug_targets_train == 1, 1, 0)))
    # print('Class 2 = ', sum(np.where(aug_targets_train == 2, 1, 0)))
    #
    # print('VALIDATION SPLIT SET')
    # print('Tot datapoints = ', aug_targets_valid.shape[0])
    # print('Class 0 = ', sum(np.where(aug_targets_valid == 0, 1, 0)))
    # print('Class 1 = ', sum(np.where(aug_targets_valid == 1, 1, 0)))
    # print('Class 2 = ', sum(np.where(aug_targets_valid == 2, 1, 0)))
    #
    # print('TEST SPLIT SET')
    # print('Tot datapoints = ', aug_targets_test.shape[0])
    # print('Class 0 = ', sum(np.where(aug_targets_test == 0, 1, 0)))
    # print('Class 1 = ', sum(np.where(aug_targets_test == 1, 1, 0)))
    # print('Class 2 = ', sum(np.where(aug_targets_test == 2, 1, 0)))
    #
    # print('************ AFTER DOWNSAMPLING + AUGMENTATION ************')
    # print('TRAIN SPLIT SET')
    # print('Tot datapoints = ', down_targets_train.shape[0])
    # print('Class 0 = ', sum(np.where(down_targets_train == 0, 1, 0)))
    # print('Class 1 = ', sum(np.where(down_targets_train == 1, 1, 0)))
    # print('Class 2 = ', sum(np.where(down_targets_train == 2, 1, 0)))
    #
    # print('VALIDATION SPLIT SET')
    # print('Tot datapoints = ', down_targets_valid.shape[0])
    # print('Class 0 = ', sum(np.where(down_targets_valid == 0, 1, 0)))
    # print('Class 1 = ', sum(np.where(down_targets_valid == 1, 1, 0)))
    # print('Class 2 = ', sum(np.where(down_targets_valid == 2, 1, 0)))
    #
    # print('TEST SPLIT SET')
    # print('Tot datapoints = ', down_targets_test.shape[0])
    # print('Class 0 = ', sum(np.where(down_targets_test == 0, 1, 0)))
    # print('Class 1 = ', sum(np.where(down_targets_test == 1, 1, 0)))
    # print('Class 2 = ', sum(np.where(down_targets_test == 2, 1, 0)))

    down_smiles_train, down_targets_train = load_train_data(os.path.join(data_dir, 'augmented_downsampled2_split_train.csv'))
    down_smiles_valid, down_targets_valid = load_train_data(os.path.join(data_dir, 'augmented_downsampled2_split_valid.csv'))
    down_smiles_test, down_targets_test = load_train_data(os.path.join(data_dir, 'augmented_downsampled2_split_test.csv'))

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

    print('TEST SPLIT SET')
    print('Tot datapoints = ', targets_test.shape[0])
    print('Class 0 = ', sum(np.where(targets_test == 0, 1, 0)))
    print('Class 1 = ', sum(np.where(targets_test == 1, 1, 0)))
    print('Class 2 = ', sum(np.where(targets_test == 2, 1, 0)))

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

    print('TEST SPLIT SET')
    print('Tot datapoints = ', aug_targets_test.shape[0])
    print('Class 0 = ', sum(np.where(aug_targets_test == 0, 1, 0)))
    print('Class 1 = ', sum(np.where(aug_targets_test == 1, 1, 0)))
    print('Class 2 = ', sum(np.where(aug_targets_test == 2, 1, 0)))

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

    print('TEST SPLIT SET')
    print('Tot datapoints = ', down_targets_test.shape[0])
    print('Class 0 = ', sum(np.where(down_targets_test == 0, 1, 0)))
    print('Class 1 = ', sum(np.where(down_targets_test == 1, 1, 0)))
    print('Class 2 = ', sum(np.where(down_targets_test == 2, 1, 0)))
    """