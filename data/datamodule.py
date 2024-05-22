from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from hydra.utils import instantiate
#import clipdataset
import torch


class DataModule:
    def __init__(
        self,
        train_dataset_path,
        real_images_val_path,
        train_transform,
        val_transform,
        batch_size,
        num_workers,
        clip=False
    ):
        print(train_dataset_path)
        if clip:
            self.dataset = clipdataset.ClipDataset(train_dataset_path, preprocess=train_transform)
        else:
            self.dataset = ImageFolder(train_dataset_path, transform=train_transform)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset,
            [
                int(0.8 * len(self.dataset)),
                len(self.dataset) - int(0.8 * len(self.dataset)),
            ],
            generator=torch.Generator().manual_seed(3407),
        )
        self.val_dataset.transform = val_transform
        if clip:
            self.real_images_val_dataset = clipdataset.ClipDataset(
                real_images_val_path, preprocess=val_transform
            )
        else:
            self.real_images_val_dataset = ImageFolder(
                real_images_val_path, transform=val_transform
            )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.idx_to_class = {v: k for k, v in self.dataset.class_to_idx.items()}

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return {
            "synthetic_val": DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            ),
            "real_val": DataLoader(
                self.real_images_val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            ),
        }
