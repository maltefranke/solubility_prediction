import logging

from data_preparation import *

from simpletransformers.classification import ClassificationModel
import sklearn
import torch


def load_our_dataset(path: str, flag_test=False, augmented=False) -> Tuple[List[str], List[str], np.array]:

    df = pd.read_csv(path)

    if not augmented:
        del df['Id']  # delete id column

    if flag_test:
        df.rename(columns={'smiles': 'text'}, inplace=True)
        return df
    else:
        df.rename(columns={'smiles': 'text', 'sol_category': 'labels'}, inplace=True)
        df = sklearn.utils.shuffle(df)
        ind_0 = np.array(df[df['labels'] == 0].index)
        ind_1 = np.array(df[df['labels'] == 1].index)
        ind_2 = np.array(df[df['labels'] == 2].index)
        ind_train = np.hstack([ind_0[0:int(0.8*ind_0.shape[0])],
                               ind_1[0:int(0.8*ind_1.shape[0])],
                               ind_2[0:int(0.8*ind_2.shape[0])]])
        ind_valid = np.hstack([ind_0[int(0.8*ind_0.shape[0]):int(0.9*ind_0.shape[0])],
                               ind_1[int(0.8*ind_1.shape[0]):int(0.9*ind_1.shape[0])],
                               ind_2[int(0.8*ind_2.shape[0]):int(0.9*ind_2.shape[0])]])
        ind_test = np.hstack([ind_0[int(0.9 * ind_0.shape[0]):],
                              ind_1[int(0.9 * ind_1.shape[0]):],
                              ind_2[int(0.9 * ind_2.shape[0]):]])
        df_train = df.iloc[ind_train]
        df_valid = df.iloc[ind_valid]
        df_test = df.iloc[ind_test]

        print(df_train.head())
        print(df_valid.head())
        print(df_test.head())
        return df_train, df_valid, df_test


if __name__ == "__main__":
    this_dir = os.path.dirname(os.getcwd())

    data_dir = os.path.join(this_dir, "data")
    # we use the augmented dataset
    train_path = os.path.join(data_dir, "augmented_smiles.csv")
    test_path = os.path.join(data_dir, "test.csv")

    train_df, valid_df, test_df = load_our_dataset(train_path, flag_test=False, augmented=True)

    # set up a logger to record if any issues occur
    # and notify us if there are any problems with the arguments we've set for the model.
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True

    model = ClassificationModel('roberta', 'seyonec/PubChem10M_SMILES_BPE_396_250', num_labels=3,
                                # weight=sample_weights[0:int(0.8*train_data_size)],
                                args={'evaluate_each_epoch': True,
                                      'evaluate_during_training_verbose': True,
                                      'no_save': True, 'num_train_epochs': 10,
                                      'auto_weights': True}, use_cuda=use_cuda)

    model.train_model(train_df, eval_df=valid_df, output_dir='chemBERTa_model', multi_label=True, num_labels=3,
                      use_cuda=use_cuda)

    # EVALUATION OF THE MODEL - wrt a test set for which we know the true predictions
    # accuracy
    print('Accuracy score')
    result, model_outputs, wrong_predictions = model.eval_model(test_df, acc=sklearn.metrics.accuracy_score)

    # Cohen-kappa evaluation
    print('Cohen-kappa metrics')

    test_df_predictions, raw_test_df_outputs = model.predict(test_df['text'].tolist())

    kappa = sklearn.metrics.cohen_kappa_score(test_df_predictions, test_df['labels'].tolist(), weights='quadratic')
    print(kappa)

    # TEST SET
    # Loading the test set
    test_dataset = load_our_dataset(test_path, flag_test=True)
    submission_ids, submission_smiles = load_test_data(test_path)

    # PREDICTIONS
    final_predictions, raw_outputs = model.predict(test_dataset['text'].tolist())

    submission_file = os.path.join(this_dir, "chemberta_augmented_predictions.csv")
    create_submission_file(submission_ids, final_predictions, submission_file)








