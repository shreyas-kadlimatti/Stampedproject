import os
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset

# ===================================================
# CROWD DATASET CLASS
# ===================================================
class CrowdDataset(Dataset):

    def __init__(
        self,
        image_dir,
        density_dir
    ):

        # ===================================================
        # DATASET PATHS
        # ===================================================
        self.image_dir = image_dir

        self.density_dir = density_dir

        # ===================================================
        # LOAD IMAGE FILES
        # ===================================================
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(".jpg")
        ])

    # ===================================================
    # TOTAL DATASET SIZE
    # ===================================================
    def __len__(self):

        return len(self.image_files)

    # ===================================================
    # GET SINGLE SAMPLE
    # ===================================================
    def __getitem__(self, idx):

        # ===================================================
        # IMAGE PATH
        # ===================================================
        img_name = self.image_files[idx]

        img_path = os.path.join(
            self.image_dir,
            img_name
        )

        # ===================================================
        # DENSITY MAP PATH
        # ===================================================
        density_name = img_name.replace(
            ".jpg",
            ".npy"
        )

        density_path = os.path.join(
            self.density_dir,
            density_name
        )

        # ===================================================
        # LOAD IMAGE
        # ===================================================
        img = cv2.imread(img_path)

        if img is None:

            raise FileNotFoundError(
                f"Image not found: {img_path}"
            )

        # ===================================================
        # BGR TO RGB
        # ===================================================
        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        # ===================================================
        # NORMALIZE IMAGE
        # ===================================================
        img = img.astype(
            np.float32
        ) / 255.0

        # ===================================================
        # LOAD DENSITY MAP
        # ===================================================
        density = np.load(density_path)

        density = density.astype(
            np.float32
        )

        # ===================================================
        # CONVERT TO TENSORS
        # ===================================================
        img = torch.tensor(
            img
        ).permute(2,0,1)

        density = torch.tensor(
            density
        ).unsqueeze(0)

        # ===================================================
        # RETURN SAMPLE
        # ===================================================
        return img, density