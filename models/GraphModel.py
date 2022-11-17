import chemprop
from data_preparation import *


# Training
def train_graph_model(data_path='../data/train.csv', CV=5, batch_size=50):
    """
    Function that trains the MPNN
    :param data_path:
    :param CV:
    :return: mean, standard deviation
    """
    arguments = [
        '--data_path', data_path,
        '--dataset_type', 'multiclass',
        '--save_dir', 'GraphModelCheckpoint',
        '--smiles_columns', 'smiles',
        '--target_columns', 'sol_category',
        "--metric", "cross_entropy",
        "--dropout", "0.05",
        "--epochs", "1",
        "--num_folds", f"{CV}",
        '--batch_size', f"{batch_size}"
    ]

    args = chemprop.args.TrainArgs().parse_args(arguments)
    mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)

    return mean_score, std_score


# Testing
def test_graph_model(data_path='../data/test.csv', preds_path='output_graphs.csv', CV=5):
    """
    Function that makes predictions on our test data starting from the trained model
    :param data_path:
    :param preds_path:
    :param CV:
    :return: final predictions
    """
    arguments = [
        '--test_path', data_path,
        '--preds_path', preds_path,
        '--checkpoint_dir', 'GraphModelCheckpoint',
        '--smiles_columns', 'smiles'
    ]

    args = chemprop.args.PredictArgs().parse_args(arguments)
    preds = chemprop.train.make_predictions(args=args)

    return preds


def load_preds_data(preds_path: str) -> Tuple[List[str], List[str], np.array, np.array, np.array]:
    """
    Function to load the output file created by test_graph_model
    :param preds_path:
    :return:
    """
    df = pd.read_csv(preds_path)

    ids = df["Id"].values.tolist()
    smiles = df["smiles"].values.tolist()
    sol0 = df["sol_category_class_0"].values.tolist()
    sol0 = np.array(sol0)
    sol1 = df["sol_category_class_1"].values.tolist()
    sol1 = np.array(sol1)
    sol2 = df["sol_category_class_2"].values.tolist()
    sol2 = np.array(sol2)

    return ids, smiles, sol0, sol1, sol2


if __name__ == "__main__":

    this_dir = os.path.dirname(os.getcwd())

    data_dir = os.path.join(this_dir, "data")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    # Train the model
    # we can change several parameters...
    mean, std = train_graph_model(batch_size=256)

    # Make predictions
    output = test_graph_model()

    # Deduce the right final predictions
    # get output of graph model
    preds_path = 'output_graphs.csv'
    submission_ids, smiles, sol0, sol1, sol2 = load_preds_data(preds_path)

    final_predictions = []

    for i in range(sol0.shape[0]):
        if sol0[i] == max(sol0[i], sol1[i], sol2[i]):
            pred_i = 0
        elif sol1[i] == max(sol0[i], sol1[i], sol2[i]):
            pred_i = 1
        elif sol2[i] == max(sol0[i], sol1[i], sol2[i]):
            pred_i = 2
        final_predictions.append(pred_i)

    final_predictions = np.array(final_predictions)

    # Are there some molecules in classes 0 and 1???
    print(np.sum(final_predictions != 2))

    # Create submission file
    submission_file = os.path.join(this_dir, "predictions_graphs.csv")
    create_submission_file(submission_ids, final_predictions, submission_file)

