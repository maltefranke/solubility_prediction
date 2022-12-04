from chemberta10M import *
import tqdm
import re

@click.command()
@click.option("--size", type=int, default=300)
@click.option("--num_classes", type=int, default=3)
@click.option("--num_layers", type=int, default=0)
@click.option("--data_dir", type=str, default="../../data/")
@click.option("--batch_size", type=int, default=30)
@click.option("--model_name", type=str, default="best.ckpt")

def main(size,
         num_classes,
         num_layers,
         data_dir,
         batch_size,
         model_name
         ):

    """
    Train and evaluate model
    """
    seed = 0
    seed_everything(seed, workers = True)

    def kappa(y, ypred):
        return cohen_kappa_score(y, ypred, weights="quadratic")

    # Load model checkpoint
    ckpt = torch.load(f"models/checkpoint/{model_name}")

    # Infer num_layers
    last_bias = list(ckpt["state_dict"].keys())[-1]
    # num_layers = int(float(re.sub(r"model.(.*).bias","\\1", last_bias))) - 1

    model = ChemBERTa(
        size=size,
        num_classes=num_classes,
        # num_layers=num_layers,
        data_dir=data_dir,
        weights=False
        )

    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.setup()
    model.to("cuda")

    valid_data = model.val_dataloader()
    test_data = model.test_dataloader()

    with torch.no_grad():
        valid_ys = []
        valid_preds = []
        for batch in tqdm.tqdm(valid_data):
            x, y, idxs, p_id = batch
            logits = model(x)
            pred = torch.argmax(nn.Softmax(dim=1)(logits),dim=1)

            valid_ys += list(y)
            valid_preds += list(pred.cpu().numpy())

        valid_kappa = kappa(valid_ys, valid_preds)

        test_preds = []
        test_ids = []
        for batch in tqdm.tqdm(test_data):
            x, idxs, p_id = batch
            logits = model(x)
            pred = torch.argmax(nn.Softmax(dim=1)(logits),dim=1)

            test_preds += list(pred.cpu().numpy())
            test_ids += list(p_id)

        test_df = pd.DataFrame({"Id": test_ids, "pred": test_preds})
        test_df.to_csv(f"submission_robertaTEST_weights_k_{valid_kappa}.csv",
                       index=False)


if __name__ == "__main__":
    main()

