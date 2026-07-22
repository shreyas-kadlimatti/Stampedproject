import os
import cv2
import torch
import random
import numpy as np

from torch.utils.data import Dataset


class CrowdDataset(Dataset):

    def __init__(self, image_dir, density_dir, augment=False):

        self.image_dir = image_dir
        self.density_dir = density_dir
        self.augment = augment

        self.image_files = [
            f for f in os.listdir(image_dir)
            if f.endswith(".jpg")
        ]

        self.image_files.sort()

    def __len__(self):

        return len(self.image_files)

    def __getitem__(self, idx):

        img_name = self.image_files[idx]

        img_path = os.path.join(self.image_dir, img_name)

        density_name = img_name.replace(".jpg", ".npy")

        density_path = os.path.join(self.density_dir, density_name)

        img = cv2.imread(img_path)

        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.0

        density = np.load(density_path).astype(np.float32)

        if self.augment:
            img, density = self.apply_augmentation(img, density)

        img = np.ascontiguousarray(img)
        density = np.ascontiguousarray(density)

        img = torch.from_numpy(img).permute(2, 0, 1).float()

        density = torch.from_numpy(density).unsqueeze(0).float()

        return img, density

    def apply_augmentation(self, img, density):

        # ---------------------------------------------------
        # 1. Horizontal Flip
        # Safe for crowd images
        # ---------------------------------------------------
        if random.random() < 0.5:
            img = np.fliplr(img)
            density = np.fliplr(density)

        # ---------------------------------------------------
        # 2. Brightness Augmentation
        # Apply only to image, not density map
        # ---------------------------------------------------
        if random.random() < 0.4:
            factor = random.uniform(0.8, 1.2)
            img = np.clip(img * factor, 0, 1)

        # ---------------------------------------------------
        # 3. Contrast Augmentation
        # Apply only to image, not density map
        # ---------------------------------------------------
        if random.random() < 0.4:
            mean = img.mean(axis=(0, 1), keepdims=True)
            factor = random.uniform(0.8, 1.2)
            img = np.clip((img - mean) * factor + mean, 0, 1)

        # ---------------------------------------------------
        # 4. Small Gaussian Noise
        # Apply only to image, not density map
        # ---------------------------------------------------
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.015, img.shape)
            img = np.clip(img + noise, 0, 1)

        return img, density