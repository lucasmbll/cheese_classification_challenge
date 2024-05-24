import data_augmentation


dir = "/Data/mellah.adib/cheese_classification_challenge/dataset/tuned_val"
data_augmentor = data_augmentation.DataAugmentation(data_dir=dir)
data_augmentor.augment_images()
