import os
import cv2
import torch
import random
import numpy as np

from torch.utils.data import Dataset


# ===================================================
# IMAGENET NORMALIZATION
# ===================================================
IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


class CrowdDataset(Dataset):

    def __init__(
        self,
        image_dir,
        density_dir,
        augment=False
    ):

        self.image_dir = image_dir
        self.density_dir = density_dir
        self.augment = augment

        if not os.path.exists(image_dir):

            raise FileNotFoundError(
                f"Image directory not found: {image_dir}"
            )

        if not os.path.exists(density_dir):

            raise FileNotFoundError(
                f"Density directory not found: {density_dir}"
            )

        self.image_files = [
            file_name
            for file_name in os.listdir(image_dir)
            if file_name.lower().endswith(".jpg")
        ]

        self.image_files.sort()

        if len(self.image_files) == 0:

            raise RuntimeError(
                f"No JPG images found in: {image_dir}"
            )

    def __len__(self):

        return len(self.image_files)

    def __getitem__(self, idx):

        img_name = self.image_files[idx]

        img_path = os.path.join(
            self.image_dir,
            img_name
        )

        density_name = (
            os.path.splitext(img_name)[0]
            + ".npy"
        )

        density_path = os.path.join(
            self.density_dir,
            density_name
        )

        # ------------------------------------------------
        # LOAD IMAGE
        # ------------------------------------------------
        img = cv2.imread(img_path)

        if img is None:

            raise FileNotFoundError(
                f"Image not found: {img_path}"
            )

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        img = img.astype(
            np.float32
        ) / 255.0

        # ------------------------------------------------
        # LOAD DENSITY MAP
        # ------------------------------------------------
        if not os.path.exists(density_path):

            raise FileNotFoundError(
                f"Density map not found: {density_path}"
            )

        density = np.load(
            density_path
        ).astype(np.float32)

        if density.ndim != 2:

            raise ValueError(
                f"Expected 2D density map, got: "
                f"{density.shape}"
            )

        if img.shape[:2] != density.shape:

            raise ValueError(
                f"Image and density sizes differ for "
                f"{img_name}. Image: {img.shape[:2]}, "
                f"Density: {density.shape}"
            )

        # ------------------------------------------------
        # AUGMENTATION
        # ------------------------------------------------
        if self.augment:

            img, density = self.apply_augmentation(
                img,
                density
            )

        # ------------------------------------------------
        # IMAGENET NORMALIZATION
        # Required for pretrained VGG16
        # ------------------------------------------------
        img = (
            img - IMAGENET_MEAN
        ) / IMAGENET_STD

        # ------------------------------------------------
        # NUMPY CONTIGUOUS ARRAYS
        # ------------------------------------------------
        img = np.ascontiguousarray(
            img,
            dtype=np.float32
        )

        density = np.ascontiguousarray(
            density,
            dtype=np.float32
        )

        # ------------------------------------------------
        # CONVERT TO TENSORS
        # ------------------------------------------------
        img = torch.from_numpy(
            img
        ).permute(
            2,
            0,
            1
        ).float()

        density = torch.from_numpy(
            density
        ).unsqueeze(
            0
        ).float()

        return img, density

    # ===================================================
    # AUGMENTATION
    # ===================================================
    def apply_augmentation(
        self,
        img,
        density
    ):

        # ------------------------------------------------
        # HORIZONTAL FLIP
        # ------------------------------------------------
        if random.random() < 0.5:

            img = np.fliplr(img)

            density = np.fliplr(density)

        # ------------------------------------------------
        # BRIGHTNESS
        # ------------------------------------------------
        if random.random() < 0.4:

            factor = random.uniform(
                0.8,
                1.2
            )

            img = np.clip(
                img * factor,
                0.0,
                1.0
            )

        # ------------------------------------------------
        # CONTRAST
        # ------------------------------------------------
        if random.random() < 0.4:

            mean = img.mean(
                axis=(0, 1),
                keepdims=True
            )

            factor = random.uniform(
                0.8,
                1.2
            )

            img = np.clip(
                (img - mean) * factor + mean,
                0.0,
                1.0
            )

        # ------------------------------------------------
        # GAUSSIAN NOISE
        # ------------------------------------------------
        if random.random() < 0.3:

            noise = np.random.normal(
                0.0,
                0.015,
                img.shape
            ).astype(np.float32)

            img = np.clip(
                img + noise,
                0.0,
                1.0
            )

        return img, density