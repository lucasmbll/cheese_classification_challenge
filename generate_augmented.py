import torch
import wandb
import hydra
from tqdm import tqdm
from generate import generate
from data_augmentation import DataAugmentation


@hydra.main(config_path="configs/generate", config_name="config")
def generate_augmented(cfg):
    # Appeler la fonction generate() de generate.py
    #generate(cfg)
    print("Data augmentation starting")
    # Initialiser l'objet DataAugmentation
    data_augmentor = DataAugmentation(data_dir=cfg.dataset_generator.output_dir)
    data_augmentor.augment_images()


if __name__ == "__main__":
    generate_augmented()
