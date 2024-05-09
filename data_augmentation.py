import os
import torchvision.utils as torchvision
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import random

class DataAugmentation:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        print("dir is " + self.data_dir)

    def augment_images(self):
        for root, dirs, files in os.walk(self.data_dir):
            for name in dirs:
                name_dir = os.path.join(root, name)
                output_dir = os.path.join(self.data_dir+"_augmented", name)
                os.makedirs(output_dir, exist_ok=True)
                transform_original = transforms.Compose([
                    transforms.ToTensor()
                ])
                transform_augmented = transforms.Compose([
                    transforms.RandomRotation(20),
                    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
                    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
                    transforms.ToTensor()
                ])
                color_jitter_transform = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                images = [file for file in os.listdir(name_dir) if file.endswith(".jpg") or file.endswith(".png")]
                for file in tqdm(images, desc=f"Processing {name}", unit="image"):
                    image_path = os.path.join(name_dir, file)
                    img = Image.open(image_path)

                    if self.is_image_black(img):
                        continue

                    original_img = transform_original(img)
                    # Sauvegarder l'image originale
                    save_path_original = os.path.join(output_dir, f'original_{file}')
                    torchvision.save_image(original_img, save_path_original)
                    # Créer et sauvegarder 4 versions transformées aléatoires
                    k = random.randint(1, 4)
                    for i in range(4):
                        augmented_img = transform_augmented(img)
                        # Appliquer l'aberration chromatique à une seule image augmentée
                        if i == k:
                            augmented_img = color_jitter_transform(augmented_img)
                        save_path_augmented = os.path.join(output_dir, f'aug_{i}_{file}')
                        torchvision.save_image(augmented_img, save_path_augmented)


    def is_image_black(self, image):
        # Convert the image to grayscale
        gray_image = image.convert('L')
        # Calculate the average pixel intensity
        average_pixel_intensity = sum(gray_image.getdata()) / len(gray_image.getdata())
        # Check if the average intensity is below a certain threshold
        return average_pixel_intensity < 1 # Adjust the threshold as needed
        
                    

# Utilisation de la classe DataAugmentation
data_augmentor = DataAugmentation(data_dir='advanced_prompts')
data_augmentor.augment_images()
