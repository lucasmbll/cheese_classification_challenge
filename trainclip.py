import torch
import torch.nn as nn
from PIL import Image
import open_clip
from transformers import AutoProcessor, AutoModel
from data.datamodule import DataModule
from ocr import load_cheese_names
import torchvision.transforms as t

class ClipModel(nn.Module):
    def __init__(self, num_classes, frozen=False, unfreeze_last_layer=True):
        super().__init__()
        self.model, self.preprocesstrain, self.preprocessval = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.num_classes = num_classes

        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, images, text):
        return self.model(images, self.tokenizer(text))


def test_openclip(img_path):
    """inspired from https://github.com/mlfoundations/open_clip"""
    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load the model and tokenizer
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model.to(device)
    # Path to the image
    # Preprocess the image
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    # Read cheese labels from file
    with open('list_of_cheese.txt', 'r') as file:
        cheese_labels = [line.strip() for line in file.readlines()]
    # Tokenize the cheese labels
    text = tokenizer(cheese_labels).to(device)

    # Compute image and text features
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Calculate the probabilities
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    # Print the label probabilities
    print("Label probabilities:", text_probs.cpu().numpy())

    # Find the label with the highest probability
    max_prob_idx = text_probs.argmax()
    max_prob = text_probs.flatten()[max_prob_idx]
    max_label = cheese_labels[max_prob_idx]

    # Print the label with the highest probability
    print("Label with highest probability:", max_label)
    print("Probability:", max_prob.item())


def test_metaclip(img_path):
    """inspired from https://github.com/facebookresearch/MetaCLIP"""

    processor = AutoProcessor.from_pretrained("facebook/metaclip-b32-400m")
    model = AutoModel.from_pretrained("facebook/metaclip-b32-400m")
    image = Image.open(img_path)
    with open('list_of_cheese.txt', 'r') as file:
        cheese_labels = [line.strip() for line in file.readlines()]
        inputs = processor(text=cheese_labels, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        text_probs = logits_per_image.softmax(dim=-1)
        print("Label probs:", text_probs)
    
    max_prob_idx = text_probs.argmax()
    max_prob = text_probs.flatten()[max_prob_idx]
    max_label = cheese_labels[max_prob_idx]

    # Print the label with the highest probability
    print("Label with highest probability:", max_label)
    print("Probability:", max_prob.item())


def train_classif_clip(cheese_names, train_dataset_path, real_images_val_path, train_transform, val_transform, batch_size=128, num_epochs=3, lr=0.0001):
    # Instantiate the DataModule
    datamodule = DataModule(train_dataset_path, real_images_val_path, train_transform, val_transform, batch_size=batch_size, num_workers=4)
    
    # Define data loader
    dataloaders = datamodule.val_dataloader()

    # Instantiate the pre-trained OpenCLIP model
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    clip_model.eval()  # Freeze OpenCLIP model

    # Define linear layer for classification
    linear_layer = nn.Linear(512, 37) 

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(linear_layer.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = tokenizer(cheese_names).to(device)
    clip_model.to(device)
    linear_layer.to(device)

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for val_set_name, val_loader in dataloaders.items():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    image_features = clip_model.encode_image(images)
                    text_features = clip_model.encode_text(text)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                # combined_features = torch.cat((image_features, text_features), dim=-1)
                outputs = linear_layer(image_features)

                # Calculate loss
                loss = criterion(outputs, labels)

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataloaders)
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

    print('Finished Training')


if __name__ == '__main__':
    
    # img_path = "testclip_brie.jpg"
    # test_metaclip(img_path)
    train_dataset_path = "/Data/mellah.adib/cheese_classification_challenge/dataset/train/gpt-prompts-2"
    real_images_val_path = "/Data/mellah.adib/cheese_classification_challenge/dataset/val"
    train_transform = t.Compose([t.Resize((224, 224)), t.ToTensor(), t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_transform = train_transform
    cheese_names = load_cheese_names("/Data/mellah.adib/cheese_classification_challenge/list_of_cheese.txt") 
    train_classif_clip(cheese_names, train_dataset_path, real_images_val_path, train_transform, val_transform)