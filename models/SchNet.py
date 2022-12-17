import os
from typing import List

import numpy as np
import torchmetrics
import pytorch_lightning as pl

import schnetpack.src.schnetpack as spk
import schnetpack.src.schnetpack.transform as trn
import torch
from ase import Atoms
from schnetpack.src.schnetpack.data import ASEAtomsData, AtomsLoader, AtomsDataModule


from data_utils import *


def prepare_schnet_data(smiles: List[str], targets: np.array, working_dir: str, dataset: str = "train",
                        batch_size: int = 512, num_workers: int = 8, pin_memory: bool = False) -> AtomsLoader:
    """
    function to prepare the schnet database
    Args:
        smiles: list of SMILES string
        targets: np.array of target values
        working_dir: path to the working directory
        dataset: name to modify the output name of the database
        batch_size: size of the dataloader batches
        num_workers: number of workers in the dataloader
        pin_memory: whether to limit memory use
    Returns:
        None
    """
    dataset_path = os.path.join(working_dir, f'./SchNet_{dataset}.db')

    if not os.path.exists(dataset_path):
        numbers, positions = smiles_to_3d(smiles)
        atoms_list = []
        for atom_numbers, positions in zip(numbers, positions):
            atoms = Atoms(numbers=np.array(atom_numbers, dtype=int), positions=np.array(positions, dtype=float))
            atoms_list.append(atoms)

        targets = [{"solubility_class": np.array([i], dtype=int)} for i in targets]

        schnet_dataset = ASEAtomsData.create(
            dataset_path,
            distance_unit='Ang',
            property_unit_dict={'solubility_class': ''}
        )
        schnet_dataset.add_systems(targets, atoms_list)

    data = AtomsDataModule(
            dataset_path,
            num_train=1.0,  # here we assume that the dataset was already split into train and val set
            num_val=0,
            num_test=0,
            num_workers=num_workers,
            batch_size=batch_size,
            # split_file=os.path.join(working_dir, "split.npz"),
            pin_memory=pin_memory,  # set to false, when not using a GPU
            property_units={'solubility_class': ''},
            transforms=[trn.ASENeighborList(cutoff=5.)]
        )
    data.prepare_data()
    data.setup()

    dataloader = data.train_dataloader()

    return dataloader


def setup_schnet(class_weights: List[float]) -> spk.task.AtomisticTask:
    """
    Setup the training task
    Args:
        class_weights: weighs the classes. to cope with class imbalance

    Returns:
        task which can be run by pytorch-lightning Trainer
    """
    cutoff = 5.
    n_atom_basis = 256

    pairwise_distance = spk.atomistic.PairwiseDistances()  # calculates pairwise distances between atoms
    radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
    schnet = spk.representation.SchNet(
        n_atom_basis=n_atom_basis, n_interactions=3,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(cutoff)
    )

    pred_solubility = spk.atomistic.Atomwise(n_in=n_atom_basis, n_out=3, n_hidden=500, n_layers=3,
                                             output_key='solubility_class')

    nn_solubility = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[pairwise_distance],
        output_modules=[pred_solubility],
        # postprocessors=[trn.CastTo64(), trn.AddOffsets('solubility_class', add_mean=False, add_atomrefs=True)]
    )

    if class_weights is not None:
        class_weights = torch.tensor(class_weights)

    output_solubility = spk.task.ModelOutput(
        name='solubility_class',
        loss_fn=torch.nn.CrossEntropyLoss(weight=class_weights),
        loss_weight=1.,
        metrics={
            "Kappa": torchmetrics.CohenKappa(num_classes=3, weights="quadratic")
        }
    )

    task = spk.task.AtomisticTask(
        model=nn_solubility,
        outputs=[output_solubility],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 1e-4},
    )

    return task


def train_schnet(task: spk.task.AtomisticTask, train_loader: AtomsLoader, val_loader: AtomsLoader,
                 working_dir: str, epochs: int = 50) -> None:
    """
    Function to do the training of SchNet
    Args:
        task: the SchNetpack task to run, defined in :func:'setup_schnet'
        train_loader:
        val_loader:
        working_dir:
        epochs: number of training epochs

    Returns:
        None
    """
    logger = pl.loggers.TensorBoardLogger(save_dir=working_dir)
    callbacks = [
        spk.train.ModelCheckpoint(
            model_path=os.path.join(working_dir, "best_inference_model"),
            save_top_k=1,
            monitor="val_loss"
        )
    ]

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        default_root_dir=working_dir,
        max_epochs=epochs,  # for testing, we restrict the number of epochs
        accelerator="auto",
        devices="auto",
    )
    trainer.fit(task, train_loader, val_loader)


def predict_schnet(smiles: List[str], working_dir: str) -> np.array:
    """
    Function to predict on a list of given SMILES, especially intended for prediction on the submission data
    Args:
        smiles: list of SMILES strings
        working_dir: working directory path, where the model and data is saved

    Returns:
        np.array of predicted classes for the given SMILES strings
    """
    # load best model
    best_model = torch.load(os.path.join(working_dir, 'best_inference_model'), map_location=torch.device("cpu"))

    # get a converter that translates atom numbers and positions to an object that schnet can predict on
    converter = spk.interfaces.AtomsConverter(neighbor_list=trn.ASENeighborList(cutoff=5.), dtype=torch.float32)
    # get atom numbers and positions for every SMILES based on rdkit estimates
    numbers, positions = smiles_to_3d(smiles)

    all_preds = []
    for atom_numbers, positions in zip(numbers, positions):
        atoms = Atoms(numbers=np.array(atom_numbers, dtype=np.int), positions=np.array(positions, dtype=np.float))
        X = converter(atoms)

        y_pred = best_model(X)
        y_pred = y_pred["solubility_class"]
        probabilities = torch.softmax(y_pred, dim=1)
        predicted_class = probabilities.argmax(dim=1).item()
        all_preds.append(predicted_class)

    all_preds = np.array(all_preds)
    return all_preds


def schnet_pipeline(data_dir: str, model_dir: str) -> None:
    """
    Tie everything together: Get data, create dataloader, setup the task, run it and make submission file
    Args:
        data_dir: path to the data directory
        model_dir: path to the model directory

    Returns:

    """

    train_path = os.path.join(data_dir, "augmented_split_train.csv")
    test_path = os.path.join(data_dir, "augmented_split_valid.csv")

    submission_path = os.path.join(model_dir, "SchNet_submission_big_net_augmented.csv")
    if os.path.exists(submission_path):
        print("Prediction have already been made!")
        return

    best_model_path = os.path.join(model_dir, 'best_inference_model')
    if not os.path.exists(best_model_path):

        ids, smiles, targets = load_train_data(train_path)

        train_loader = prepare_schnet_data(smiles, targets, model_dir, dataset="train_augmented")
        val_loader = prepare_schnet_data(smiles, targets, model_dir, dataset="val_augmented")

        class_weights = calculate_class_weights(targets)

        task = setup_schnet(class_weights)

        train_schnet(task, train_loader, val_loader, model_dir, epochs=20)

    else:
        print("Model has already been trained!")

    submission_ids, submission_smiles = load_test_data(test_path)

    submission_preds = predict_schnet(submission_smiles, model_dir)

    create_submission_file(submission_ids, submission_preds, submission_path)


if __name__ == "__main__":
    this_dir = os.path.dirname(os.getcwd())

    data_dir = os.path.join(this_dir, "data")

    schnet_model_dir = os.path.join(os.getcwd(), "SchNet_models")
    if not os.path.exists(schnet_model_dir):
        os.mkdir(schnet_model_dir)

    schnet_pipeline(data_dir, schnet_model_dir)


