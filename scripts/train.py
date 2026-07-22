import sys
import os

# ===================================================
# ADD PROJECT ROOT
# ===================================================
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models.csrnet import CSRNet
from scripts.dataset_loader import CrowdDataset

# ===================================================
# DEVICE
# ===================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ===================================================
# PROJECT ROOT
# ===================================================
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

# ===================================================
# DATASET PATHS
# ===================================================
image_dir = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "ShanghaiTech",
    "part_A",
    "train_data",
    "images"
)

density_dir = os.path.join(
    PROJECT_ROOT,
    "density_maps"
)

# ===================================================
# SAVE PATHS
# ===================================================
checkpoint_dir = os.path.join(
    PROJECT_ROOT,
    "checkpoints"
)

os.makedirs(checkpoint_dir, exist_ok=True)

final_model_path = os.path.join(
    checkpoint_dir,
    "final_csrnet.pth"
)

print("Images Path Exists:", os.path.exists(image_dir))
print("Density Path Exists:", os.path.exists(density_dir))

# ===================================================
# LOAD DATASET WITH AUGMENTATION
# ===================================================
dataset = CrowdDataset(
    image_dir,
    density_dir,
    augment=True
)

# ===================================================
# TEST ONE SAMPLE BEFORE TRAINING
# ===================================================
sample_img, sample_den = dataset[0]

print("Sample Image Shape:", sample_img.shape)
print("Sample Density Shape:", sample_den.shape)
print("Sample Density Sum:", sample_den.sum().item())

# ===================================================
# DATALOADER
# ===================================================
dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    pin_memory=True if torch.cuda.is_available() else False
)

print("Dataset Size:", len(dataset))

# ===================================================
# MODEL
# ===================================================
model = CSRNet().to(device)

# ===================================================
# LOSS FUNCTION
# ===================================================
criterion = nn.MSELoss()

# ===================================================
# OPTIMIZER
# ===================================================
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-5,
    weight_decay=5e-4
)

# ===================================================
# TRAINING SETTINGS
# ===================================================
epochs = 25

# ===================================================
# TRAINING LOOP
# ===================================================
for epoch in range(epochs):

    print(f"\n========== Epoch {epoch + 1}/{epochs} ==========")

    running_loss = 0.0
    running_mae = 0.0

    model.train()

    for i, (images, density_maps) in enumerate(dataloader):

        images = images.to(device, non_blocking=True)
        density_maps = density_maps.to(device, non_blocking=True)

        # -------------------------------------------
        # FORWARD PASS
        # -------------------------------------------
        outputs = model(images)

        # -------------------------------------------
        # RESIZE GT DENSITY MAP WITH COUNT PRESERVATION
        # -------------------------------------------
        original_sum = density_maps.sum()

        density_maps = F.interpolate(
            density_maps,
            size=outputs.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        resized_sum = density_maps.sum()

        density_maps = density_maps * (
            original_sum / (resized_sum + 1e-8)
        )

        # -------------------------------------------
        # LOSS
        # -------------------------------------------
        loss = criterion(outputs, density_maps)

        # -------------------------------------------
        # BACKPROPAGATION
        # -------------------------------------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # -------------------------------------------
        # COUNT ERROR
        # -------------------------------------------
        pred_count = torch.relu(outputs).sum().item()
        gt_count = density_maps.sum().item()
        abs_error = abs(pred_count - gt_count)

        running_loss += loss.item()
        running_mae += abs_error

        # -------------------------------------------
        # DEBUG INFO
        # -------------------------------------------
        if (i + 1) % 10 == 0:

            print(
                f"Step [{i + 1}/{len(dataloader)}] | "
                f"Loss: {loss.item():.6f} | "
                f"GT Count: {gt_count:.2f} | "
                f"Pred Count: {pred_count:.2f} | "
                f"Abs Error: {abs_error:.2f}"
            )

    epoch_loss = running_loss / len(dataloader)
    epoch_mae = running_mae / len(dataloader)

    print(f"\nEpoch Loss: {epoch_loss:.6f}")
    print(f"Epoch MAE: {epoch_mae:.2f}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ===================================================
    # SAVE CHECKPOINT EVERY 5 EPOCHS
    # ===================================================
    if (epoch + 1) % 5 == 0:

        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"csrnet_epoch_{epoch + 1}.pth"
        )

        torch.save(
            model.state_dict(),
            checkpoint_path
        )

        latest_path = os.path.join(
            checkpoint_dir,
            "latest_csrnet.pth"
        )

        torch.save(
            model.state_dict(),
            latest_path
        )

        print(f"Checkpoint Saved: {checkpoint_path}")

# ===================================================
# SAVE FINAL MODEL
# ===================================================
torch.save(
    model.state_dict(),
    final_model_path
)

print("\nModel Saved Successfully!")
print("Saved To:", final_model_path)
# import numpy as np

# density = np.load(r"E:\Stampede\density_maps\IMG_1.npy")

# print("Min:", density.min())
# print("Max:", density.max())
# print("Sum:", density.sum())
# print("Mean:", density.mean())