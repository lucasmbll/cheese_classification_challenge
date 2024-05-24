import data_augmentation
import os


dir = "/Data/mellah.adib/cheese_classification_challenge/dataset/train_db"
data_augmentor = data_augmentation.DataAugmentation(data_dir=dir)
data_augmentor.augment_images()
