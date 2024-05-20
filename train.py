import torch
import wandb
import hydra
from tqdm import tqdm
import clip


@hydra.main(config_path="configs/train", config_name="config")
def train(cfg):
    # logger = wandb.init(entity="lucas_mbll", project="challenge_cheese", name=cfg.experiment_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    loss_txt = hydra.utils.instantiate(cfg.loss_fn)    # for openclip
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    if cfg.model.instance._target_ == "models.openclip.OpenClipTest":
        datamodule = hydra.utils.instantiate(cfg.datamodule, clip=True)
        datamodule.train_transform = model.preprocesstrain
        datamodule.val_transform = model.preprocessval

    train_loader = datamodule.train_dataloader()
    val_loaders = datamodule.val_dataloader()

    for epoch in tqdm(range(cfg.epochs)):
        model.train()
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0
        for i, batch in enumerate(train_loader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            if (cfg.model.instance._target_=="models.openclip.OpenClipTest"):
                texts = labels.to(device)
                print(texts)
                print(labels)
                logits_per_image, logits_per_text = model(images, texts)
                ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
                loss = (loss_fn(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
                loss.backward()
                if device == "cpu":
                    optimizer.step()
                else : 
                    clip.convert_models_to_fp32(model)
                    optimizer.step()
                    clip.model.convert_weights(model)
            else:
                preds = model(images)
                loss = loss_fn(preds, labels)
                # logger.log({"loss": loss.detach().cpu().numpy()})
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            epoch_loss += loss.detach().cpu().numpy() * len(images)
            epoch_num_correct += (
                (preds.argmax(1) == labels).sum().detach().cpu().numpy()
            )
            num_samples += len(images)
        epoch_loss /= num_samples
        epoch_acc = epoch_num_correct / num_samples
        """logger.log(
            {
                "epoch": epoch,
                "train_loss_epoch": epoch_loss,
                "train_acc": epoch_acc,
            }
        )"""

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
                    if cfg.model.instance._target_=="models.openclip.OpenClipTest":
                        texts = model.tokenizer(labels).to(device)
                        logits_per_image, logits_per_text = model(images, texts)
                        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
                        loss = (loss_fn(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
                        preds = logits_per_image
                    else:
                        preds = model(images)
                        loss = loss_fn(preds, labels)
                y_true.extend(labels.detach().cpu().tolist())
                y_pred.extend(preds.argmax(1).detach().cpu().tolist())
                epoch_loss += loss.detach().cpu().numpy() * len(images)
                epoch_num_correct += (preds.argmax(1) == labels).sum().detach().cpu().numpy()
                num_samples += len(images)
                epoch_loss += loss.detach().cpu().numpy() * len(images)
                epoch_num_correct += (
                    (preds.argmax(1) == labels).sum().detach().cpu().numpy()
                )
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

        """logger.log(
            {
                "epoch": epoch,
                **val_metrics,
            }
        )"""
    torch.save(model.state_dict(), cfg.checkpoint_path)


if __name__ == "__main__":
    train()

