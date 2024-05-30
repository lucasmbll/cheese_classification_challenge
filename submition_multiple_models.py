import hydra
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_pil_image
import os
from PIL import Image
import pandas as pd
import torch
import ocr
import logging
import torch.nn.functional as F

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
    logging.getLogger().setLevel(logging.ERROR)

    test_loader = DataLoader(
        TestDataset(
            cfg.dataset.test_path, hydra.utils.instantiate(cfg.dataset.test_transform)
        ),
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
    )
    
    # Load models and their checkpoints
    models = []
    base_path = os.path.join(hydra.utils.get_original_cwd(), "checkpoints")  # "checkpoints" folder in the root directory
    weights_path = ["DINOV2LARGE_dbfinalsetaug_doubletune.pt", "second_final_model.pt", "Final_balanced_lucas.pt", "Final_lucas.pt"]
    for path in weights_path:
        full_path = os.path.join(base_path, path)
        model = hydra.utils.instantiate(cfg.model.instance).to(device)
        checkpoint = torch.load(full_path)
        print(f"Loading model from checkpoint: {full_path}")
        model.load_state_dict(checkpoint)
        model.eval()  # Set model to evaluation mode
        models.append(model)

    class_names = sorted(os.listdir(cfg.dataset.train_path))
    reader = ocr.initialize_ocr(cfg.ocr_method)

    # Create submission.csv
    submission = pd.DataFrame(columns=["id", "label"])
    ocr_identified = 0
    cheese_names = ocr.load_cheese_names(os.path.join(hydra.utils.get_original_cwd(),'cheese_ocr.txt'))

    nb_images = 0
    for i, batch in enumerate(test_loader):
        nb_images += len(batch[0])
        images, image_names = batch
        images = images.to(device)
        
        # Aggregate probabilities from each model
        all_probs = torch.zeros((images.size(0), len(class_names)), device=device)
        for model in models:
            with torch.no_grad():
                prob = model(images)
                prob = F.softmax(prob, 1)
                all_probs += prob
        
        # Average the probabilities
        avg_probs = all_probs / len(models)
        preds = avg_probs.argmax(1)
        preds = [class_names[pred] for pred in preds.cpu().numpy()]

        """for j, image_name in enumerate(image_names):
            # Load the original image
            image_path = os.path.join(cfg.dataset.test_path, f"{image_name}.jpg")
            original_image = Image.open(image_path).convert('RGB')
            
            lab = ocr.classify_image(original_image, reader, cheese_names, cfg.threshold_ocr, increment=cfg.increment, ocr_method=cfg.ocr_method, comparison_method=cfg.comparison_method)
            if lab:
                preds[j] = lab
                ocr_identified += 1"""

        submission = pd.concat(
            [
                submission,
                pd.DataFrame({"id": image_names, "label": preds}),
            ]
        )
    submission.to_csv(f"{cfg.root_dir}/submission.csv", index=False)
    print(f"OCR identified {ocr_identified} labels")

if __name__ == "__main__":
    create_submission()
