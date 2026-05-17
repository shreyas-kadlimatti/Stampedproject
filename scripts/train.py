# import sys
# import os

# # ===================================================
# # ADD PROJECT ROOT
# # ===================================================
# sys.path.append(
#     os.path.abspath(
#         os.path.join(os.path.dirname(__file__), "..")
#     )
# )
import os
import sys

# ===================================================
# PROJECT ROOT
# ===================================================
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

# ===================================================
# ADD ROOT TO PYTHON PATH
# ===================================================
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models.csrnet import CSRNet
from scripts.dataset_loader import CrowdDataset

# ===================================================
# DEVICE
# ===================================================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)

# ===================================================
# DATASET PATHS
# ===================================================
image_dir = os.path.join(
    BASE_DIR,
    "dataset",
    "ShanghaiTech",
    "part_A",
    "train_data",
    "images"
)

density_dir = os.path.join(
    BASE_DIR,
    "density_maps"
)
# ===================================================
# LOAD DATASET
# ===================================================
dataset = CrowdDataset(
    image_dir,
    density_dir
)

# ===================================================
# DATALOADER
# ===================================================
dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True
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
    lr=1e-5
)

# ===================================================
# TRAINING LOOP
# ===================================================
epochs = 50

for epoch in range(epochs):

    print(f"\n========== Epoch {epoch+1}/{epochs} ==========")

    running_loss = 0.0

    model.train()

    for i, (images, density_maps) in enumerate(dataloader):

        # -------------------------------------------
        # MOVE TO DEVICE
        # -------------------------------------------
        images = images.to(device)

        density_maps = density_maps.to(device)

        # -------------------------------------------
        # FORWARD PASS
        # -------------------------------------------
        outputs = model(images)

        # -------------------------------------------
        # RESIZE GT DENSITY MAP
        # -------------------------------------------
        original_sum = density_maps.sum()

        density_maps = F.interpolate(
            density_maps,
            size=outputs.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        # -------------------------------------------
        # PRESERVE TOTAL COUNT
        # VERY IMPORTANT
        # -------------------------------------------
        resized_sum = density_maps.sum()

        if resized_sum > 0:
            density_maps = density_maps * (
                original_sum / resized_sum
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

        running_loss += loss.item()

        # -------------------------------------------
        # DEBUG INFO
        # -------------------------------------------
        if (i+1) % 10 == 0:

            pred_count = outputs.sum().item()

            gt_count = density_maps.sum().item()

            print(
                f"Step [{i+1}/{len(dataloader)}] | "
                f"Loss: {loss.item():.6f} | "
                f"GT Count: {gt_count:.2f} | "
                f"Pred Count: {pred_count:.2f}"
            )


# ===================================================
# EPOCH LOSS
# ===================================================
epoch_loss = running_loss / len(dataloader)

print(f"\nEpoch Loss: {epoch_loss:.6f}")

# ===================================================
# SAVE CHECKPOINT AFTER EVERY EPOCH
# ===================================================
checkpoint_path = os.path.join(
    BASE_DIR,
    f"csrnet_epoch_{epoch+1}.pth"
)

torch.save(
    model.state_dict(),
    checkpoint_path
)

print("Checkpoint Saved:", checkpoint_path)

# ===================================================
# SAVE TO GOOGLE DRIVE
# ===================================================
drive_checkpoint_path = f"/content/drive/MyDrive/csrnet_epoch_{epoch+1}.pth"

torch.save(
    model.state_dict(),
    drive_checkpoint_path
)

print("Drive Backup Saved:", drive_checkpoint_path)

# ===================================================
# SAVE MODEL
# ===================================================
save_path = os.path.join(
    BASE_DIR,
    "csrnet.pth"
)

torch.save(
    model.state_dict(),
    save_path
)

print("\nModel Saved Successfully!")

print("Saved To:", save_path)