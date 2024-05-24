import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import os
import torchvision.transforms as t
from ocr import load_cheese_names

def get_score_for_true_label(image, cheese_labels, true_label):
    """inspired from https://github.com/facebookresearch/MetaCLIP"""

    processor = AutoProcessor.from_pretrained("facebook/metaclip-b32-400m")
    model = AutoModel.from_pretrained("facebook/metaclip-b32-400m")
    inputs = processor(text=cheese_labels, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        text_probs = logits_per_image.softmax(dim=-1)

    # Get the index of the true label in the cheese_labels list
    true_label_idx = cheese_labels.index(true_label)
    # Get the probability score for the true label
    true_label_prob = text_probs[0, true_label_idx].item()

    return true_label_prob


if __name__ == '__main__':
    real_images_val_path = "/Data/mellah.adib/cheese_classification_challenge/dataset/val"
    cheese_names = load_cheese_names("/Data/mellah.adib/cheese_classification_challenge/list_of_cheese.txt")

    # Loop through each cheese folder
    for cheese_label in os.listdir(real_images_val_path):
        cheese_folder = os.path.join(real_images_val_path, cheese_label)
        if os.path.isdir(cheese_folder):
            # Loop through each image in the cheese folder
            for image_name in os.listdir(cheese_folder):
                image_path = os.path.join(cheese_folder, image_name)
                if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    image = Image.open(image_path)

                    # Get the score for the true label
                    true_label_prob = get_score_for_true_label(image, cheese_names, cheese_label)

                    # Print the result
                    print(f"Image: {image_name}")
                    print(f"True Label: {cheese_label}")
                    print(f"Probability for True Label: {true_label_prob}")
                    print("-" * 30)
