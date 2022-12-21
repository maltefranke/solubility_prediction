import pytorch_lightning as pl
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.metrics import cohen_kappa_score

from transformers import AutoTokenizer, RobertaForSequenceClassification
from torch.utils.data import Dataset

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (LearningRateMonitor,
                                         EarlyStopping,
                                         ModelCheckpoint,
                                         TQDMProgressBar)
from pytorch_lightning import seed_everything
import wandb
import click


def kappa(y, yhat):
    y = y.cpu().numpy()
    yhat = yhat.cpu().numpy()
    return cohen_kappa_score(y, yhat, weights="quadratic")


class SmilesDataset(Dataset):
    def __init__(self,
                 filename,
                 load_labels=True
                 ):

        self.load_labels = load_labels

        # Contains columns: Id, smiles, sol_category
        self.df = pd.read_csv(filename)

        self.smiles = (self.df["smiles"].values.tolist())

        if self.load_labels:
            self.labels = self.df["sol_category"].values
        self.point_id = self.df["Id"].values

    # Need to override methods __len__ and __getitem__
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            smiles = self.smiles[idx]
            if "Id" in self.df.columns:
                ids = self.point_id[idx]
                if self.load_labels:
                    labels = torch.as_tensor(self.labels[idx])
                    return smiles, labels, idx, ids
                else:
                    return smiles, idx, ids
            else:
                if self.load_labels:
                    labels = torch.as_tensor(self.labels[idx])
                    return smiles, labels, idx
                else:
                    return smiles, idx


class ChemBERTa(pl.LightningModule):
    def __init__(self,
                 size,
                 num_classes,
                 data_dir,
                 learning_rate=1e-3,
                 batch_size=300,
                 dropout=0.3,
                 weights=True,
                 file_template="split_{}.csv",
                 ):
        super().__init__()

        # Define loss function:
        if weights:
            print("*************************************************************")
            print("*************** training with weighted loss *****************")
            print("*************************************************************")
            self.Loss = nn.CrossEntropyLoss(weight=torch.Tensor([0.01, 0.4, 0.7]),
                                            reduction='mean')
        else:
            print("*** training WITHOUT weights")
            self.Loss = nn.CrossEntropyLoss(reduction='mean')

        # Data loading variables
        self.num_workers = 4*torch.cuda.device_count()
        self.batch_size = batch_size

        # Data paths
        self.data_dir = data_dir
        self.train_file = file_template.format("train")
        self.valid_file = file_template.format("valid")
        self.test_file = "test.csv"

        # Model specific variables
        self.learning_rate = learning_rate

        # Define PyTorch model
        self.pretrained = "DeepChem/ChemBERTa-10M-MTR" #DeepChem/ChemBERTa-77M-MTR
        self.tokenizer = (AutoTokenizer.
                          from_pretrained(
                              self.pretrained
                          ))
        self.model = (RobertaForSequenceClassification
                      .from_pretrained(
                          self.pretrained,
                          num_labels=num_classes
                      ))

    def forward(self, x):
        # define prediction/inference actions
        x = self.tokenizer(list(x),
                           return_tensors="pt",
                           padding=True)

        x = {key: x[key].to("cuda:0")
             for key in x.keys()}

        return self.model(**x).logits

    def training_step(self, batch, batch_idx):
        # define train loop
        x, y, idxs, p_ids = batch
        logits = self(x)

        loss = self.Loss(logits, y)

        self.log(f"train_loss", loss, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, idxs, p_ids = batch
        logits = self(x)

        pred = nn.Softmax(dim=1)(logits)
        pred = torch.argmax(pred, dim=1)
        kap = kappa(y, pred)
        self.log(f"valid_kappa", kap, on_epoch=True, on_step=False, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, idxs, p_ids = batch
        logits = self(x)
        pred = nn.Softmax(dim=1)(logits)
        pred = torch.argmax(pred, dim=1).cpu().numpy()

        return pd.DataFrame(list(zip(p_ids, pred)))

    def test_epoch_end(self, outputs):
        # Concat all test results
        print(outputs)
        all_outs = pd.concat(outputs)
        print(all_outs)
        all_outs.columns = ["Id", "pred"]
        all_outs.to_csv(f"Chemberta_train.csv", index=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.learning_rate)

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode="max",
                factor=0.3,
                patience=1,
                cooldown=0,
                verbose=True
            ),
            "monitor": "valid_kappa"
            }
        return [optimizer], [lr_scheduler]

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_data = SmilesDataset(self.data_dir + self.train_file)
            self.val_data = SmilesDataset(self.data_dir + self.valid_file)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_data = SmilesDataset(self.data_dir + self.test_file,
                                           load_labels=False)

    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data,
                          batch_size=2000,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data,
                          batch_size=2000,
                          num_workers=self.num_workers)

#############################################################

@click.command()
@click.option("--size", type=int, default=300)
@click.option("--num_classes", type=int, default=3)
@click.option("--max_epochs", type=int, default=50)
@click.option("--data_dir", type=str, default="../../data/")
@click.option("--learning_rate", type=float, default=1e-3)
@click.option("--batch_size", type=int, default=30)
@click.option("--weights", is_flag=True)
def main(size,
         num_classes,
         max_epochs,
         data_dir,
         learning_rate,
         batch_size,
         weights
         ):

    """
    Train and evaluate model
    """
    seed = 0
    seed_everything(seed, workers=True)

    wandb.init(project="solubility_prediction")

    model = ChemBERTa(
        size=size,
        num_classes=num_classes,
        data_dir=data_dir,
        learning_rate=learning_rate,
        batch_size=batch_size,
        weights=weights
    )

    wandb_logger = WandbLogger()
    wandb.watch(model)

    checkpoint_callback = ModelCheckpoint(dirpath="models/checkpoint/",
                                          filename="best",
                                          save_last=False,
                                          save_top_k=1,
                                          monitor="valid_kappa",
                                          mode="max")

    earlystop_callback = EarlyStopping(monitor="valid_kappa",
                                       mode="max",
                                       patience=3,
                                       min_delta=0.001,
                                       verbose=True)

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=max_epochs,
        callbacks=[TQDMProgressBar(refresh_rate=5),
                   LearningRateMonitor(logging_interval="epoch"),
                   #earlystop_callback,
                   checkpoint_callback,
                   ],
        logger=wandb_logger,
        deterministic=True
    )


    # Train
    trainer.fit(model)
    # Save model
    torch.save(model.state_dict(), 'models/checkpoint/last_weights.pth')

    # Test model
    trainer.test(ckpt_path="best")



if __name__ == "__main__":
    main()
