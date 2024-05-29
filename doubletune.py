import torch
import wandb
import hydra
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

@hydra.main(config_path="configs/train", config_name="config", version_base=None)
def train(cfg):
    logger = wandb.init(entity="lucas_mbll", project="challenge_cheese", name=cfg.experiment_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    checkpoint = torch.load("/Data/mellah.adib/cheese_classification_challenge/checkpoints/DINOV2LARGE_dbfinalsetaug.pt")
    print(f"Loading model from checkpoint: /Data/mellah.adib/cheese_classification_challenge/checkpoints/DINOV2LARGE_dbfinalsetaug.pt")
    model.load_state_dict(checkpoint)
    
    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    train_loader = datamodule.train_dataloader()
    val_loaders = datamodule.val_dataloader()

    scheduler = None
    if cfg.get("scheduler"):  # Check if scheduler is specified in the config
        scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)

    best_val_acc = 0.0
    best_model_path = "/Data/mellah.adib/cheese_classification_challenge/checkpoints/DINOV2LARGE_dbfinalsetaug_doubletune.pt"
    for epoch in tqdm(range(cfg.epochs)):
        model.train()
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0
        for i, batch in enumerate(train_loader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)
            loss = loss_fn(preds, labels)
            logger.log({"loss": loss.detach().cpu().numpy()})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().cpu().numpy() * len(images)
            epoch_num_correct += (preds.argmax(1) == labels).sum().detach().cpu().numpy()
            num_samples += len(images)

        if scheduler:
            if cfg.scheduler._target_ == 'torch.optim.lr_scheduler.ReduceLROnPlateau':
                scheduler.step(epoch_loss)
            else:
                scheduler.step()

        epoch_loss /= num_samples
        epoch_acc = epoch_num_correct / num_samples
        logger.log(
            {
                "epoch": epoch,
                "train_loss_epoch": epoch_loss,
                "train_acc": epoch_acc,
            }
        )

        val_metrics = {}
        model.eval()
        for val_set_name, val_loader in val_loaders.items():
            epoch_loss = 0
            epoch_num_correct = 0
            num_samples = 0
            y_true = []
            y_pred = []
            for i, batch in enumerate(val_loader):
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                with torch.no_grad():
                    preds = model(images)
                    loss = loss_fn(preds, labels)

                y_true.extend(labels.detach().cpu().tolist())
                y_pred.extend(preds.argmax(1).detach().cpu().tolist())
                epoch_loss += loss.detach().cpu().numpy() * len(images)
                epoch_num_correct += (preds.argmax(1) == labels).sum().detach().cpu().numpy()
                num_samples += len(images)

            epoch_loss /= num_samples
            epoch_acc = epoch_num_correct / num_samples
            val_metrics[f"{val_set_name}/loss"] = epoch_loss
            val_metrics[f"{val_set_name}/acc"] = epoch_acc
            val_metrics[f"{val_set_name}/confusion_matrix"] = (
                wandb.plot.confusion_matrix(
                    y_true=y_true,
                    preds=y_pred,
                    class_names=[
                        datamodule.idx_to_class[i][:10].lower()
                        for i in range(len(datamodule.idx_to_class))
                    ],
                )
            )

        logger.log(
            {
                "epoch": epoch,
                **val_metrics,
            }
        )

        # Check if the current model has the best real_val accuracy
        if f"real_val/acc" in val_metrics and val_metrics["real_val/acc"] > best_val_acc:
            best_val_acc = val_metrics["real_val/acc"]
            torch.save(model.state_dict(), best_model_path)

    if best_model_path:
        print(f"Best model saved at: {best_model_path}")

if __name__ == "__main__":
    train()
