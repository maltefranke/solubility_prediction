import os
import chemprop


def fit_CGR(data_path, save_dir, CV=5):
    arguments = ["--data_path",  f"{data_path}",
                 "--dataset_type",  "multiclass",
                 "--target_columns", "target",
                 "--metric", "crossentropy",
                 "--dropout", "0.05",
                 "--epochs", "300",
                 "--num_folds",  f"{CV}",
                 "--save_dir", f"{save_dir}"]

    args = chemprop.args.TrainArgs().parse_args(arguments)
    mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
    return mean_score, std_score

