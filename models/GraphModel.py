import chemprop
import numpy as np
import tempfile
import csv

from data_preparation import *


# Training
def train_graph_model(data_path: str, ensemble_size=5, batch_size=128, class_weights=None):
    ids, smiles, targets = load_train_data(data_path)

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(["class_weights"])
        for target in targets:
            if class_weights is None:
                writer.writerow([1])
            else:
                writer.writerow([class_weights[int(target)]])

    arguments = [
        '--data_path', data_path,
        '--smiles_columns', 'smiles',
        '--target_columns', 'sol_category',
        '--dataset_type', 'multiclass',
        '--data_weights_path', f'{csvfile.name}',
        '--multiclass_num_classes', "3",
        '--save_dir', 'GraphModels',
        "--metric", "cross_entropy",
        "--dropout", "0.1",
        "--epochs", "50",
        "--ensemble_size", f"{ensemble_size}",
        '--batch_size', f"{batch_size}"
    ]

    args = chemprop.args.TrainArgs().parse_args(arguments)
    mae, std = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
    metrics_log = {"mae": mae, "std": std}
    return metrics_log


# Testing
def predict_graph_model(data_path: str):
    arguments = [
        '--test_path', data_path,
        '--preds_path', "/dev/null",
        '--checkpoint_dir', 'GraphModels',
        '--smiles_columns', 'smiles'
    ]

    args = chemprop.args.PredictArgs().parse_args(arguments)
    preds = chemprop.train.make_predictions(args=args)
    preds = np.array(preds).squeeze()
    print(preds)
    print(preds.shape)

    # this takes the argument (class) of the highest probability
    predictions = np.argmax(preds, axis=1)

    print(predictions.shape)
    return predictions


def graph_pipeline(data_dir: str, model_dir: str):

    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    graph_dir = os.path.join(model_dir, "GraphModels")

    submission_path = os.path.join(model_dir, "graph_submission.csv")
    if os.path.exists(submission_path):
        print("Prediction have already been made!")

    else:
        # assume training has been done when the graph_dir exists
        if not os.path.exists(graph_dir):
            train_log = train_graph_model(train_path)

        submission_preds = predict_graph_model(test_path)

        submission_ids, _ = load_test_data(test_path)

        # Create submission file
        submission_file = os.path.join(graph_dir, "predictions_graphs.csv")
        create_submission_file(submission_ids, submission_preds, submission_file)


if __name__ == "__main__":

    this_dir = os.path.dirname(os.getcwd())

    data_dir = os.path.join(this_dir, "data")
    model_dir = os.getcwd()

    create_subsample_train_csv(data_dir)

    graph_pipeline(data_dir, model_dir)

