import chemprop
import numpy as np
import tempfile
import csv
import h5py

from data_utils import *


# Training
def train_graph_model(data_path: str, ensemble_size=10, batch_size=256):
    ids, smiles, targets = load_train_data(data_path)

    # write a class weight temp csv file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(["class_weights"])
        for target in targets:
            class_weights = calculate_class_weights(targets)
            writer.writerow([class_weights[int(target)]])

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as qm_features:
        npy_file = qm_features.name + ".npy"
        # qm_descriptors = smiles_to_qm_descriptors(smiles, data_dir=os.path.dirname(data_path))
        descriptor_file = os.path.join(os.path.dirname(data_path), "random_undersampling_descriptors.h5")
        with h5py.File(descriptor_file, "r") as hf:
            qm_descriptors = hf["descriptors"][:]
        qm_descriptors, mean, std = standardize(qm_descriptors)
        # qm_descriptors, columns_info, standardization_data = nan_imputation(qm_descriptors, 0.5)
        # print(qm_descriptors.shape)
        np.save(npy_file, qm_descriptors, allow_pickle=True)


    arguments = [
        '--data_path', data_path,
        '--seed', '13',
        '--pytorch_seed', '13',
        '--smiles_columns', 'smiles',
        '--target_columns', 'sol_category',
        '--dataset_type', 'multiclass',
        '--features_generator', 'rdkit_2d',
        #'--data_weights_path', f'{csvfile.name}',
        '--multiclass_num_classes', "3",
        #'--features_path', npy_file,
        '--save_dir', 'GraphModels',
        "--metric", "cross_entropy",
        '--extra_metrics', 'f1',
        '--aggregation', 'sum',
        # '--split_type', 'scaffold_balanced',
        '--hidden_size', '256',
        '--ffn_num_layers', '5',
        '--depth', '5',
        "--dropout", "0",
        "--epochs", "500",
        "--ensemble_size", f"{ensemble_size}",
        '--batch_size', f"{batch_size}"
    ]

    args = chemprop.args.TrainArgs().parse_args(arguments)
    mae, std = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
    metrics_log = {"mae": mae, "std": std}
    breakpoint()
    return metrics_log


# Testing
def predict_graph_model(data_path: str):

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as qm_features:
        npy_file = qm_features.name + ".npy"
        qm_descriptors = smiles_to_qm_descriptors(_, data_dir=os.path.dirname(data_path), type_="test")
        np.save(npy_file, qm_descriptors, allow_pickle=True)

    arguments = [
        '--test_path', data_path,
        '--preds_path', "/dev/null",
        '--checkpoint_dir', 'GraphModels',
        '--smiles_columns', 'smiles',
        '--features_path', npy_file,
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

    train_path = os.path.join(data_dir, "random_undersampling.csv")
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
    smiles = 0
    qm_descriptors = smiles_to_qm_descriptors(smiles, data_dir)

    create_subsample_train_csv(data_dir, qm_descriptors)

    graph_pipeline(data_dir, model_dir)

