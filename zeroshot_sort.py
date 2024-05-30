import os
import torch
import open_clip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


"""Created to conduct data curation. Not used due to a lack of efficiency."""

# Define your zero-shot scoring function
def score_zeroshot(image, cheese, cheese_labels, score_lim=0.1):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    image = preprocess(image).unsqueeze(0).to(device)

    # Tokenize the cheese labels
    text = tokenizer(cheese_labels).to(device)

    # Compute image and text features
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    score = text_probs.flatten()[cheese_labels.index(cheese)]
    return score.item()

# Path to the dataset folder
dataset_dir = r"C:\Users\adib4\OneDrive\Documents\Travail\X\MODAL DL\cheese_classification_challenge\scp_copy\dreambooth4"  # Replace with your dataset path
cheese_labels = [folder for folder in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, folder))]

# Dictionary to store scores for each label
scores_dict = {cheese: [] for cheese in cheese_labels}

# Iterate through the dataset
for cheese in cheese_labels:
    cheese_folder = os.path.join(dataset_dir, cheese)
    for img_name in os.listdir(cheese_folder):
        img_path = os.path.join(cheese_folder, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
            score = score_zeroshot(image, cheese, cheese_labels)
            scores_dict[cheese].append(score)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

# Calculate and print statistics for each cheese label
for cheese, scores in scores_dict.items():
    if scores:
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        q1 = np.percentile(scores, 25)
        q3 = np.percentile(scores, 75)
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        print(f"Statistics for {cheese}:")
        print(f"  Mean: {mean_score}")
        print(f"  Median: {median_score}")
        print(f"  Q1: {q1}")
        print(f"  Q3: {q3}")
        print(f"  Min: {min_score}")
        print(f"  Max: {max_score}")
        print(f"  Number of images: {len(scores)}")
        
        # Optionally, plot the score distribution
        plt.hist(scores, bins=20, alpha=0.75, edgecolor='black')
        plt.title(f"Score Distribution for {cheese}")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.75)
        plt.show()

    else:
        print(f"No scores available for {cheese}.")
