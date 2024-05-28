import torch
import hydra
from tqdm import tqdm
import warnings
import optuna
import logging
from hydra.utils import instantiate
from omegaconf import OmegaConf
import pandas as pd
import os
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def objective(trial, cfg):
    # Define the hyperparameter search space
    lr = trial.suggest_loguniform('optim.lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('datamodule.batch_size', [16, 32, 64, 128])
    beta1 = trial.suggest_uniform('optim.betas[0]', 0.8, 0.999)
    beta2 = trial.suggest_uniform('optim.betas[1]', 0.8, 0.999)
    weight_decay = trial.suggest_loguniform('optim.weight_decay', 1e-5, 1e-2)
    scheduler_choice = trial.suggest_categorical('scheduler', ['null', 'steplr', 'plateau'])

    scheduler_params = {}
    if scheduler_choice == 'steplr':
        scheduler_params['step_size'] = trial.suggest_int('scheduler.steplr.step_size', 1, 10)
        scheduler_params['gamma'] = trial.suggest_uniform('scheduler.steplr.gamma', 0.05, 0.5)
    elif scheduler_choice == 'plateau':
        scheduler_params['factor'] = trial.suggest_uniform('scheduler.plateau.factor', 0.1, 0.9)
        scheduler_params['patience'] = trial.suggest_int('scheduler.plateau.patience', 1, 10)

    # Update the configuration with the hyperparameters
    cfg.optim.lr = lr
    cfg.datamodule.batch_size = batch_size
    cfg.optim.betas = (beta1, beta2)
    cfg.optim.weight_decay = weight_decay

    # Train the model with the updated configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    optimizer = instantiate(cfg.optim, params=model.parameters())
    loss_fn = instantiate(cfg.loss_fn)
    datamodule = instantiate(cfg.datamodule)

    train_loader = datamodule.train_dataloader()
    val_loaders = datamodule.val_dataloader()

    scheduler = None
    if scheduler_choice == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
    elif scheduler_choice == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)

    best_val_acc = 0.0
    train_acc_per_epoch = []
    val_acc_per_epoch = []

    # Log the parameters used for the current trial
    logger.info(f"Trial {trial.number}: Parameters - {trial.params}")

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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().cpu().numpy() * len(images)
            epoch_num_correct += (preds.argmax(1) == labels).sum().detach().cpu().numpy()
            num_samples += len(images)

        if scheduler:
            if scheduler_choice == 'plateau':
                scheduler.step(epoch_loss)
            else:
                scheduler.step()

        epoch_loss /= num_samples
        epoch_acc = epoch_num_correct / num_samples
        train_acc_per_epoch.append(epoch_acc)
        logger.info(f"Epoch {epoch}: train_loss_epoch = {epoch_loss}, train_acc = {epoch_acc}")

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

            if val_set_name == 'real_val':
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc

        val_acc_per_epoch.append(val_metrics['real_val/acc'])
        logger.info(f"Epoch {epoch}: val_metrics = {val_metrics}")

    # Save trial results to CSV
    results = {
        'trial_number': trial.number,
        'best_val_acc': best_val_acc,
        'train_acc_per_epoch': train_acc_per_epoch,
        'val_acc_per_epoch': val_acc_per_epoch,
        'params': trial.params
    }

    results_df = pd.DataFrame([results])
    if not os.path.isfile('trial_results.csv'):
        results_df.to_csv('trial_results.csv', index=False)
    else:
        results_df.to_csv('trial_results.csv', mode='a', header=False, index=False)

    # Return only the value needed for Optuna optimization
    return best_val_acc

@hydra.main(config_path="configs/train", config_name="config", version_base=None)
def main(cfg):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, cfg), n_trials=20)
    
    logger.info(f"Best trial: {study.best_trial.value}")
    logger.info(f"Best hyperparameters: {study.best_trial.params}")
    
    # Save the study results
    df = study.trials_dataframe()
    df.to_csv("optuna_results.csv", index=False)

    # Plotting training and validation accuracies per epoch for each trial
    for i, trial in enumerate(study.trials):
        train_acc_per_epoch = trial.user_attrs['train_acc_per_epoch']
        val_acc_per_epoch = trial.user_attrs['val_acc_per_epoch']
        
        plt.plot(train_acc_per_epoch, label=f'Trial {i+1}: Train Acc')
        plt.plot(val_acc_per_epoch, label=f'Trial {i+1}: Val Acc')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracies per Epoch')
    plt.legend()
    plt.savefig('accuracy_per_epoch.png')
    plt.show()

    # Analysis of results
    logger.info("Analysis of Results:")
    logger.info("====================")
    logger.info("The best hyperparameters found:")
    logger.info(study.best_trial.params)
    logger.info(f"The best validation accuracy found: {study.best_trial.value}")

if __name__ == "__main__":
    main()