import os
from typing import List

import numpy as np
import torchmetrics
import pytorch_lightning as pl

import schnetpack.src.schnetpack as spk
import schnetpack.src.schnetpack.transform as trn
import torch
from ase import Atoms
from schnetpack.src.schnetpack.data import ASEAtomsData
from schnetpack.src.schnetpack.data import AtomsDataModule

from data_preparation import *


def prepare_schnet_data(smiles: List[str], targets: np.array, working_dir: str,
                        dataset: str="train") -> AtomsDataModule:
    """
    function to prepare the schnet database
    Args:
        numbers: list of int atom numbers for each molecule
        positions: list of float atom positions for each molecule
        targets: np.array of target values
        dataset: name to modify the output name of the database

    Returns:
        None
    """
    dataset_path = os.path.join(working_dir, f'./SchNet_{dataset}_for_testing.db')

    if not os.path.exists(dataset_path):
        numbers, positions = smiles_to_3d(smiles)
        atoms_list = []
        for atom_numbers, positions in zip(numbers, positions):
            atoms = Atoms(numbers=np.array(atom_numbers, dtype=np.int), positions=np.array(positions, dtype=np.float))
            # atoms = Atoms(numbers=atom_numbers)
            atoms_list.append(atoms)

        targets = [{"solubility_class": np.array([i], dtype=np.int)} for i in targets]

        schnet_dataset = ASEAtomsData.create(
            dataset_path,
            distance_unit='Ang',
            property_unit_dict={'solubility_class': ''}
        )
        schnet_dataset.add_systems(targets, atoms_list)

    data = AtomsDataModule(
            dataset_path,
            batch_size=10,
            num_train=0.8,
            num_val=0.1,
            num_test=0.1,
            num_workers=1,
            split_file=os.path.join(working_dir, "split_for_testing.npz"),
            pin_memory=False,  # set to false, when not using a GPU
            property_units={'solubility_class': ''},
            transforms=[trn.ASENeighborList(cutoff=5.)]
            #transforms=[trn.ASENeighborList(cutoff=5.),trn.RemoveOffsets("solubility_class", remove_mean=False, remove_atomrefs=True), # trn.CastTo32() ],
        )
    data.prepare_data()
    data.setup()

    return data


def setup_schnet(class_weights: List[float]) -> spk.task.AtomisticTask:

    cutoff = 5.
    n_atom_basis = 30

    pairwise_distance = spk.atomistic.PairwiseDistances()  # calculates pairwise distances between atoms
    radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
    schnet = spk.representation.SchNet(
        n_atom_basis=n_atom_basis, n_interactions=3,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(cutoff)
    )

    pred_solubility = spk.atomistic.Atomwise(n_in=n_atom_basis, n_out=3, output_key='solubility_class')

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
        optimizer_args={"lr": 1e-4}
    )

    return task


def train_schnet(task: spk.task.AtomisticTask, data: AtomsDataModule, working_dir: str, epochs: int = 50):
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
    )
    trainer.fit(task, datamodule=data)


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
    best_model = torch.load(os.path.join(working_dir, 'best_inference_model'))

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


if __name__ == "__main__":
    this_dir = os.path.dirname(os.getcwd())

    data_dir = os.path.join(this_dir, "data")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    schnet_model_dir = os.path.join(os.getcwd(), "SchNet models")
    if not os.path.exists(schnet_model_dir):
        os.mkdir(schnet_model_dir)

    # get data and transform smiles -> morgan fingerprint
    ids, smiles, targets = load_train_data(train_path)

    data = prepare_schnet_data(smiles[:100], targets[:100], schnet_model_dir)

    class_weights = calculate_class_weights(targets)

    task = setup_schnet(class_weights)

    train_schnet(task, data, schnet_model_dir, epochs=3)

    predict_schnet(smiles[100:150], schnet_model_dir)

