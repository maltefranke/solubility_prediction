import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# TODO refine ANN architecture
class ClassificationNeuralNetwork(torch.nn.Module):
    """
    A standard ANN
    """

    def __init__(self, input_dim: int, output_dim: int = 2, hidden_dim=300, dropout_rate=0.3):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.net = torch.nn.Sequential(
                        torch.nn.Linear(input_dim, hidden_dim),
                        torch.nn.ReLU(),
                        self.dropout,
                        torch.nn.Linear(hidden_dim, output_dim))

    def forward(self, x):

        out = self.net(x)

        return out

    def predict_class(self, x):
        
        with torch.no_grad:
            logits = self(x)
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
def train(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, optimizer, device) -> float:
    model.train()
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    loss_all = torch.tensor(0)
    count = 0

    for data in train_loader:
        fp = data.fps.to(device)
        target = data.targets.to(device)

        model.zero_grad()

        temp_real_values = target.squeeze()

        temp_predictions = model(fp).reshape((temp_real_values.shape))

        # TODO which loss should we take?
        loss = loss_fn(temp_predictions, temp_real_values)
        loss.backward()
        loss_all += loss

        optimizer.step()

        count += list(temp_real_values.size())[0]

    return loss_all.item() / count


def test(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, device: str) -> float:
    model.eval()
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    loss_all = torch.tensor(0)
    count = 0

    for data in test_loader:
        fp = data.fps.to(device)
        target = data.targets.to(device)

        temp_real_values = target.squeeze()
        temp_predictions = model(fp).reshape((temp_real_values.shape))

        # TODO which loss should we take?
        loss = loss_fn(temp_predictions, temp_real_values)
        loss_all += loss

        count += list(temp_real_values.size())[0]

    return loss_all.item() / count


class EarlyStopping:
    # adapted from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=10, verbose=False, delta=0, save_path='', model_id=0):
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
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
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
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        save_model_weights = os.path.join(self.save_path, f'model{self.model_id}')
        torch.save(model.state_dict(), save_model_weights)
        self.val_loss_min = val_loss


def train_ann(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader,
              epochs: int, lr: float, es_patience: int, save_path: str, model_id: int = 0):

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=es_patience, verbose=True, save_path=save_path, model_id=model_id)

    train_losses = []
    test_losses = []
    best_test_loss = None

    for epoch in range(epochs):

        print(f"Start epoch {epoch}...")
        train_loss = train(model, train_loader, optimizer, device).item()
        print("Train loss: ", train_loss)

        test_loss = test(model, test_loader, device).item()
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
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train')
    plt.plot(range(1, len(test_errors) + 1), test_errors, label='Test')

    # mark the early stopping epoch
    plt.axvline(es_epoch, linestyle='--', color='r', label='Early Stopping Epoch')

    plt.xlabel('Epochs')
    plt.ylabel('Mean absolute error')
    plt.grid(True)
    plt.legend(frameon=False)
    plt.tight_layout()
    save_plot = os.path.join(save_path, 'loss')
    plt.savefig(save_plot, bbox_inches='tight')
    plt.close()
