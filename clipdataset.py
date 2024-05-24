from PIL import Image
from torch.utils.data import Dataset
import os
import open_clip

class ClipDataset(Dataset):
    def __init__(self, root_dir, preprocess):
        self.root_dir = root_dir
        self.preprocess = preprocess
        self.img_paths = []
        self.labels = []
        self.class_to_idx = {}  # Manually create label to index mapping
        self.load_data()

    def load_data(self):
        idx = 0
        for label_folder in os.listdir(self.root_dir):
            label_path = os.path.join(self.root_dir, label_folder)
            if os.path.isdir(label_path):
                self.class_to_idx[label_folder] = idx
                idx += 1
                for img_file in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_file)
                    self.img_paths.append(img_path)
                    self.labels.append(label_folder)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = self.preprocess(Image.open(img_path))
        label = self.labels[idx]
        label_idx = self.class_to_idx[label]  # Get the index for the label
        return image, label_idx  # Return label index instead of label


if __name__ == '__main__':
    ROOT_DIR = r"C:\Users\adib4\OneDrive\Documents\Travail\X\MODAL DL\cheese_classification_challenge\dataset\train\firstGen"
    # Assuming you have a preprocess function for CLIP, replace it below
    _, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    dataset = ClipDataset(ROOT_DIR, preprocess)
    print("Total images:", len(dataset))
    print("Sample data:", dataset[739])  # Sample data format: (image, caption)
