import hydra
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_pil_image
import os
from PIL import Image
import pandas as pd
import torch
import ocr
import logging
logging.getLogger().setLevel(logging.ERROR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestDataset(Dataset):
    def __init__(self, test_dataset_path, test_transform):
        self.test_dataset_path = test_dataset_path
        self.test_transform = test_transform
        images_list = os.listdir(self.test_dataset_path)
        # filter out non-image files
        self.images_list = [image for image in images_list if image.endswith(".jpg")]

    def __getitem__(self, idx):
        image_name = self.images_list[idx]
        image_path = os.path.join(self.test_dataset_path, image_name)
        image = Image.open(image_path)
        image = self.test_transform(image)
        return image, os.path.splitext(image_name)[0]

    def __len__(self):
        return len(self.images_list)


@hydra.main(config_path="configs/train", config_name="config", version_base=None)
def create_submission(cfg):
    test_loader = DataLoader(
        TestDataset(
            cfg.dataset.test_path, hydra.utils.instantiate(cfg.dataset.test_transform)
        ),
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
    )
    # Load model and checkpoint
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    path = "/Data/mellah.adib/cheese_classification_challenge/checkpoints/DINOV2_gpt_prompts.pt"
    checkpoint = torch.load(path)
    #checkpoint = torch.load(cfg.checkpoint_path)
    print(f"Loading model from checkpoint: {cfg.checkpoint_path}")
    model.load_state_dict(checkpoint)
    class_names = sorted(os.listdir(cfg.dataset.train_path))
    reader = ocr.initialize_ocr(cfg.ocr_method)

    # Create submission.csv
    submission = pd.DataFrame(columns=["id", "label"])

    for i, batch in enumerate(test_loader):
        images, image_names = batch
        images = images.to(device)
        preds = model(images)
        preds = preds.argmax(1)
        preds = [class_names[pred] for pred in preds.cpu().numpy()]
        """for i, image in enumerate(images):
            # Convert PyTorch tensor to PIL image
            im = to_pil_image(image.to('cpu'))
            # Convert PIL image to RGB mode
            im = im.convert('RGB')
            lab = ocr.classify_image(im, reader, [])
            if lab: 
                print(preds[i], lab)
                preds[i] = lab
                print(f"OCR detected label: {lab} for image: {image_names[i]}")"""

        submission = pd.concat(
            [
                submission,
                pd.DataFrame({"id": image_names, "label": preds}),
            ]
        )
    submission.to_csv(f"{cfg.root_dir}/submission.csv", index=False)


if __name__ == "__main__":
    create_submission()
