import os
import torchvision.transforms as transforms
import torchvision.utils as torchvision
from PIL import Image
from tqdm import tqdm
import random
import cv2
import numpy as np

class DataAugmentation:
    def __init__(self, data_dir, aggressive_augmentations=0):
        self.data_dir = data_dir
        self.aggressive_augmentations = aggressive_augmentations
        print("dir is " + self.data_dir)

    def augment_images(self):
        for root, dirs, files in os.walk(self.data_dir):
            for name in dirs:
                name_dir = os.path.join(root, name)
                output_dir = os.path.join(self.data_dir+"_augmented", name)
                os.makedirs(output_dir, exist_ok=True)
                
                # Original and augmented transformations
                transform_original = transforms.Compose([
                    transforms.ToTensor()
                ])

                # Apply regular or aggressive augmentations based on the flag
                if self.aggressive_augmentations==1:
                    transform_augmented = transforms.Compose([
                        transforms.RandomChoice([
                            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.05),
                            transforms.ElasticTransform(alpha=100.0),
                            # Add more aggressive transformations here
                        ]),
                        transforms.RandomRotation(30),
                        transforms.RandomResizedCrop(256, scale=(0.7, 1.0)),
                        transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.7, 1.3), shear=20),
                        transforms.RandomPerspective(distortion_scale=0.3),
                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.7),
                        transforms.ToTensor(),
                        transforms.RandomApply([transforms.RandomErasing()], p=0.5)
                    ])
                elif self.aggressive_augmentations==2:
                    transform_augmented = transforms.Compose([
                        transforms.RandomChoice([
                            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                            transforms.ElasticTransform(alpha=150.0),
                            # Add more aggressive transformations here
                        ]),
                        transforms.RandomRotation(50),
                        transforms.RandomResizedCrop(256, scale=(0.6, 1.0)),
                        transforms.RandomAffine(degrees=50, translate=(0.3, 0.3), scale=(0.6, 1.3), shear=30),
                        transforms.RandomPerspective(distortion_scale=0.4),
                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.8),
                        transforms.ToTensor(),
                        transforms.RandomApply([transforms.RandomErasing()], p=0.4)
                    ])
                else:
                    transform_augmented = transforms.Compose([
                        transforms.RandomChoice([
                            transforms.RandomRotation(10),
                            transforms.RandomResizedCrop(256, scale=(0.9, 1.0)),
                            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5),
                            transforms.RandomPerspective(distortion_scale=0.1),
                            # Add more regular transformations here
                        ]),
                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
                        transforms.ToTensor()
                    ])
                
                images = [file for file in os.listdir(name_dir) if file.endswith(".jpg") or file.endswith(".png")]
                for file in tqdm(images, desc=f"Processing {name}", unit="image"):
                    image_path = os.path.join(name_dir, file)
                    img = Image.open(image_path)

                    if self.is_image_black(img):
                        continue

                    original_img = transform_original(img)
                    # Save the original image
                    save_path_original = os.path.join(output_dir, f'original_{file}')
                    torchvision.save_image(original_img, save_path_original)

                    nb_images = 2
                    if self.aggressive_augmentations==1:
                        nb_images = 10
                    elif self.aggressive_augmentations==2:
                        nb_images = 20
                    elif self.aggressive_augmentations>=3:
                        nb_images = 10*self.aggressive_augmentations

                    for i in range(nb_images):
                        augmented_img = transform_augmented(img)
                        save_path_augmented = os.path.join(output_dir, f'aug_{i}_{file}')
                        torchvision.save_image(augmented_img, save_path_augmented)


    def is_image_black(self, image):
        # Convert the image to grayscale
        gray_image = image.convert('L')
        # Calculate the average pixel intensity
        average_pixel_intensity = sum(gray_image.getdata()) / len(gray_image.getdata())
        # Check if the average intensity is below a certain threshold
        return average_pixel_intensity < 1 # Adjust the threshold as needed





"""
                transform_augmented = transforms.Compose([
                    transforms.RandomChoice([
                        transforms.RandomRotation(20),
                        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
                        transforms.RandomPerspective(distortion_scale=0.2),
                        # Add more creative transformations here
                    ]),
"""