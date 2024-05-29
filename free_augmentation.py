import torch
from tqdm import tqdm
from generate import generate
from data_augmentation import DataAugmentation


def generate_augmented(path):
    # Appeler la fonction generate() de generate.py
    #generate(cfg)
    print("Data augmentation starting")
    # Initialiser l'objet DataAugmentation
    data_augmentor = DataAugmentation(data_dir=path, aggressive_augmentations=1)
    data_augmentor.augment_images()


if __name__ == "__main__":
    path = r"C:\Users\adib4\OneDrive\Documents\Travail\X\MODAL DL\cheese_classification_challenge\scp_copy/doubletune_dino"
    generate_augmented(path)
