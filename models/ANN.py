import os
from typing import List
import numpy as np
import torch
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data_utils import *
from conversion_smiles_utils import *


class ClassificationNeuralNetwork(torch.nn.Module):
    """
    A standard ANN
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 2,
        hidden_dim: int = 300,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            self.dropout,
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            self.dropout,
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):

        out = self.net(x)

        return out

    def predict_class(self, x):

        with torch.no_grad:
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
            predicted_classes = probabilities.argmax(dim=1)

        return predicted_classes


class FingerprintData(torch.utils.data.Dataset):
    def __init__(self, fps, targets):
        self.fps = fps
        self.targets = targets

    def __len__(self):
        return self.fps.shape[0]

    def __getitem__(self, idx):
        fp = self.fps[idx]
        targets = self.targets[idx]
        return fp, targets


# training epoch
def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer,
    device: str,
    label_weights: List[float] = None,
) -> float:
    model.train()
    model.to(device)

    if label_weights is not None:
        label_weights = torch.tensor(label_weights).to(device)
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=label_weights, reduction="mean"
        )
    else:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

    loss_all = torch.tensor(0).type(torch.float).to(device)
    num_batches = 0

    for data in train_loader:
        fp = data.fps.to(device)
        target = data.targets.type(torch.LongTensor).to(device)

        model.zero_grad()

        temp_predictions = model(fp).reshape(target.shape[0], -1)

        # calculates the mean loss of the batch
        batch_loss = loss_fn(temp_predictions, target)
        batch_loss.backward()
        loss_all += batch_loss

        optimizer.step()

        num_batches += 1

    # loss = sum(batchlosses) / num_batches
    return loss_all.item() / num_batches


def test(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str,
) -> float:
    model.eval()
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    loss_all = torch.tensor(0).type(torch.float).to(device)
    num_batches = 0

    for data in test_loader:
        fp = data.fps.to(device)
        target = data.targets.type(torch.LongTensor).to(device)

        temp_predictions = model(fp).reshape(target.shape[0], -1)

        batch_loss = loss_fn(temp_predictions, target)
        loss_all += batch_loss

        num_batches += 1

    return loss_all.item() / num_batches


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(
        self, patience=10, verbose=False, delta=0, save_path="", model_id=0
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path
        self.model_id = model_id

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decreases.
        """

        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        save_model_weights = os.path.join(
            self.save_path, f"model{self.model_id}"
        )
        torch.save(model.state_dict(), save_model_weights)
        self.val_loss_min = val_loss


def train_ann(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    epochs: int,
    lr: float,
    es_patience: int,
    save_path: str,
    model_id: int = 0,
    label_weights: List[float] = None,
):

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(
        patience=es_patience,
        verbose=True,
        save_path=save_path,
        model_id=model_id,
    )

    train_losses = []
    test_losses = []
    best_test_loss = None

    for epoch in range(epochs):

        print(f"Start epoch {epoch}...")
        train_loss = train(
            model, train_loader, optimizer, device, label_weights
        )
        print("Train loss: ", train_loss)

        test_loss = test(model, test_loader, device)
        print("Test loss: ", test_loss)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if best_test_loss is None or test_loss < best_test_loss:
            best_epoch = epoch
            best_test_loss = test_loss

        early_stopping(test_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    print("Training finished!")

    return train_losses, test_losses, best_epoch, best_test_loss


def plotting(train_losses, test_errors, es_epoch, save_path):
    train_losses = [t for t in train_losses]
    test_errors = [t for t in test_errors]

    # Plot train, val, test details and save plot
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train")
    plt.plot(range(1, len(test_errors) + 1), test_errors, label="Test")

    # mark the early stopping epoch
    plt.axvline(
        es_epoch, linestyle="--", color="r", label="Early Stopping Epoch"
    )

    plt.xlabel("Epochs")
    plt.ylabel("Cross-Entropy loss")
    plt.grid(True)
    plt.legend(frameon=False)
    plt.tight_layout()
    save_plot = os.path.join(save_path, "loss")
    plt.savefig(save_plot, bbox_inches="tight")
    plt.close()


def checkpoints_exist(ann_save_path: str, CV: int) -> bool:
    for model_id in range(CV):
        model_path = os.path.join(
            ann_save_path, str(model_id), f"model{model_id}"
        )
        if not os.path.exists(model_path):
            return False
    return True


def get_checkpoints(ann_save_path: str, CV: int):
    assert os.path.exists(
        ann_save_path
    ), "Run training before calling this function"

    model_checkpoints = []
    for model_id in range(CV):
        model_path = os.path.join(
            ann_save_path, str(model_id), f"model{model_id}"
        )
        if os.path.exists(model_path):
            temp_params = torch.load(model_path)
            model_checkpoints.append(temp_params)

    return model_checkpoints


def ann_learning(
    X,
    y,
    ann_save_path=None,
    CV=5,
    hidden_dim=300,
    dropout_rate=0.3,
    epochs=300,
    lr=1e-3,
    es_patience=50,
    label_weights: List[float] = None,
) -> list:

    assert ann_save_path is not None, (
        "Please set a ann_save_path inside of your fit_fn_parameters! The ANN weights will"
        "be saved under this path"
    )
    if not os.path.exists(ann_save_path):
        os.mkdir(ann_save_path)

    # if checkpoints exist, load and return them
    if checkpoints_exist(ann_save_path, CV):
        model_checkpoints = get_checkpoints(ann_save_path, CV)
        return model_checkpoints

    input_dim = X.shape[-1]
    output_dim = 3  # y.shape[-1] doesn't work because it's not one hot encoded

    best_epochs, best_test_losses = [], []

    kfold = KFold(n_splits=CV, shuffle=True)

    for i, (train_idx, test_idx) in enumerate(kfold.split(X)):
        ann_model = ClassificationNeuralNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
        )

        X_train_i, X_test_i = X[train_idx], X[test_idx]
        y_train_i, y_test_i = y[train_idx], y[test_idx]

        train_data = FingerprintData(X_train_i, y_train_i)
        test_data = FingerprintData(X_test_i, y_test_i)

        def collate_tuple(data):
            fps = [
                torch.from_numpy(i[0].astype(np.float32)).reshape(1, -1)
                for i in data
            ]
            fps = torch.cat(fps, dim=0)
            targets = [i[1].astype(np.float32) for i in data]
            targets = torch.tensor(targets)

            data_object = FingerprintData(fps=fps, targets=targets)
            return data_object

        train_dataloader = DataLoader(
            train_data, batch_size=64, shuffle=True, collate_fn=collate_tuple
        )
        test_dataloader = DataLoader(
            test_data, batch_size=64, shuffle=True, collate_fn=collate_tuple
        )

        model_dir = os.path.join(ann_save_path, str(i))
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        train_loss, test_loss, best_epoch, best_test_loss = train_ann(
            ann_model,
            train_dataloader,
            test_dataloader,
            epochs,
            lr,
            es_patience,
            model_dir,
            i,
            label_weights,
        )

        best_epochs.append(best_epoch)
        best_test_losses.append(best_test_loss)

        plotting(train_loss, test_loss, best_epoch, model_dir)

    model_checkpoints = get_checkpoints(ann_save_path, CV)

    return model_checkpoints


def predict_ann_ensemble(
    X,
    model_checkpoints: list,
    output_dim: int = 3,
    hidden_dim: int = 300,
) -> np.array:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    X = torch.tensor(X).type(torch.FloatTensor).to(device)

    input_dim = X.shape[-1]

    predictions = []

    for model_checkpoint in model_checkpoints:
        net = ClassificationNeuralNetwork(
            input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim
        )
        net.load_state_dict(model_checkpoint)
        net = net.to(device)
        net.eval()

        # get logits
        model_prediction = net(X)
        predictions.append(model_prediction.reshape(-1, 1, output_dim))

    predictions = torch.cat(predictions, dim=1)

    final_predictions = torch.sum(predictions[:], dim=1)

    final_predictions = final_predictions.argmax(dim=1).cpu().numpy()

    return final_predictions


if __name__ == "__main__":
    this_dir = os.getcwd()
    root_dir = os.path.dirname(this_dir)

    data_dir = os.path.join(root_dir, "data")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    ids, smiles, targets = load_train_data(train_path)
    all_fps = smiles_to_morgan_fp(smiles)

    label_weights = calculate_class_weights(targets)

    model_checkpoints = ann_learning(all_fps, targets, ann_save_path=os.path.join(this_dir, "ANNResults"),
                                     label_weights=label_weights)

    submission_ids, submission_smiles = load_test_data(test_path)

    submission_fps = smiles_to_morgan_fp(submission_smiles)

    predictions = predict_ann_ensemble(submission_fps, model_checkpoints)

    submission_file = os.path.join(root_dir, "submissions", "ANN_predictions.csv")
    create_submission_file(submission_ids, predictions, submission_file)
