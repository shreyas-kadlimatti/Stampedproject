import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset

# ===================================================
# ADD PROJECT ROOT
# ===================================================
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.append(PROJECT_ROOT)

from models.csrnet import CSRNet
from scripts.dataset_loader import CrowdDataset


# ===================================================
# SETTINGS
# ===================================================
SEED = 42

EPOCHS = 50
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 5e-4

VALIDATION_RATIO = 0.10

COUNT_LOSS_WEIGHT = 1.0
GRADIENT_CLIP_VALUE = 5.0

NUM_WORKERS = 0


# ===================================================
# RANDOM SEED
# ===================================================
def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ===================================================
# DEVICE
# ===================================================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

use_amp = device.type == "cuda"

print("Using Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


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
# CHECKPOINT PATHS
# ===================================================
checkpoint_dir = os.path.join(
    PROJECT_ROOT,
    "checkpoints"
)

os.makedirs(checkpoint_dir, exist_ok=True)

best_model_path = os.path.join(
    checkpoint_dir,
    "best_csrnet.pth"
)

final_model_path = os.path.join(
    checkpoint_dir,
    "final_csrnet.pth"
)

latest_checkpoint_path = os.path.join(
    checkpoint_dir,
    "latest_csrnet_checkpoint.pth"
)


# ===================================================
# VERIFY PATHS
# ===================================================
print("Images Path Exists:", os.path.exists(image_dir))
print("Density Path Exists:", os.path.exists(density_dir))

if not os.path.exists(image_dir):
    raise FileNotFoundError(
        f"Training image folder not found: {image_dir}"
    )

if not os.path.exists(density_dir):
    raise FileNotFoundError(
        f"Density map folder not found: {density_dir}"
    )


# ===================================================
# CREATE TRAIN AND VALIDATION DATASETS
# ===================================================
# Separate instances are required so validation does
# not use random augmentation.
train_dataset_full = CrowdDataset(
    image_dir,
    density_dir,
    augment=True
)

validation_dataset_full = CrowdDataset(
    image_dir,
    density_dir,
    augment=False
)


# ===================================================
# TRAIN / VALIDATION SPLIT
# ===================================================
dataset_size = len(train_dataset_full)

indices = np.arange(dataset_size)

random_generator = np.random.default_rng(SEED)
random_generator.shuffle(indices)

validation_size = max(
    1,
    int(dataset_size * VALIDATION_RATIO)
)

validation_indices = indices[:validation_size].tolist()
train_indices = indices[validation_size:].tolist()

train_dataset = Subset(
    train_dataset_full,
    train_indices
)

validation_dataset = Subset(
    validation_dataset_full,
    validation_indices
)


# ===================================================
# TEST ONE SAMPLE
# ===================================================
sample_img, sample_density = train_dataset[0]

print("Sample Image Shape:", sample_img.shape)
print("Sample Density Shape:", sample_density.shape)
print("Sample Density Sum:", sample_density.sum().item())

print("Training Images:", len(train_dataset))
print("Validation Images:", len(validation_dataset))


# ===================================================
# DATA LOADERS
# ===================================================
pin_memory = torch.cuda.is_available()

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory
)


# ===================================================
# MODEL
# ===================================================
# This loads ImageNet-pretrained VGG16 frontend weights.
model = CSRNet(
    pretrained=True
).to(device)


# ===================================================
# LOSS FUNCTIONS
# ===================================================
density_criterion = nn.MSELoss()

count_criterion = nn.SmoothL1Loss()


# ===================================================
# OPTIMIZER
# ===================================================
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ===================================================
# LEARNING RATE SCHEDULER
# ===================================================
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
    min_lr=1e-7
)


# ===================================================
# MIXED PRECISION
# ===================================================
scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)


# ===================================================
# RESIZE DENSITY MAP WHILE PRESERVING COUNT
# ===================================================
def resize_density_map(density_maps, output_size):

    original_sum = density_maps.sum(
        dim=(1, 2, 3),
        keepdim=True
    )

    resized_density = F.interpolate(
        density_maps,
        size=output_size,
        mode="bilinear",
        align_corners=False
    )

    resized_sum = resized_density.sum(
        dim=(1, 2, 3),
        keepdim=True
    )

    resized_density = resized_density * (
        original_sum / (resized_sum + 1e-8)
    )

    return resized_density


# ===================================================
# COMBINED LOSS
# ===================================================
def calculate_loss(outputs, targets):

    density_loss = density_criterion(
        outputs,
        targets
    )

    predicted_counts = outputs.sum(
        dim=(1, 2, 3)
    )

    ground_truth_counts = targets.sum(
        dim=(1, 2, 3)
    )

    # Scaling prevents large crowd counts from making
    # count loss dominate density-map loss.
    count_loss = count_criterion(
        predicted_counts / 1000.0,
        ground_truth_counts / 1000.0
    )

    total_loss = (
        density_loss
        + COUNT_LOSS_WEIGHT * count_loss
    )

    return (
        total_loss,
        density_loss,
        count_loss,
        predicted_counts,
        ground_truth_counts
    )


# ===================================================
# SAVE FULL TRAINING CHECKPOINT
# ===================================================
def save_training_checkpoint(
    path,
    epoch,
    best_validation_mae
):

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_validation_mae": best_validation_mae
        },
        path
    )


# ===================================================
# TRAINING
# ===================================================
best_validation_mae = float("inf")

for epoch in range(1, EPOCHS + 1):

    print(
        f"\n========== Epoch {epoch}/{EPOCHS} =========="
    )

    # =================================================
    # TRAINING PHASE
    # =================================================
    model.train()

    training_loss_sum = 0.0
    training_mae_sum = 0.0

    for step, (images, density_maps) in enumerate(
        train_loader,
        start=1
    ):

        images = images.to(
            device,
            non_blocking=True
        )

        density_maps = density_maps.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # ---------------------------------------------
        # MIXED-PRECISION FORWARD PASS
        # ---------------------------------------------
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp
        ):

            outputs = model(images)

            targets = resize_density_map(
                density_maps,
                outputs.shape[-2:]
            )

            (
                total_loss,
                density_loss,
                count_loss,
                predicted_counts,
                ground_truth_counts
            ) = calculate_loss(
                outputs,
                targets
            )

        # ---------------------------------------------
        # BACKPROPAGATION
        # ---------------------------------------------
        scaler.scale(
            total_loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP_VALUE
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        # ---------------------------------------------
        # COUNT ERROR
        # ---------------------------------------------
        absolute_error = torch.abs(
            predicted_counts.detach()
            - ground_truth_counts
        ).mean().item()

        training_loss_sum += total_loss.item()
        training_mae_sum += absolute_error

        # ---------------------------------------------
        # DEBUG OUTPUT
        # ---------------------------------------------
        if step % 10 == 0 or step == len(train_loader):

            print(
                f"Step [{step}/{len(train_loader)}] | "
                f"Total Loss: {total_loss.item():.6f} | "
                f"Density Loss: {density_loss.item():.6f} | "
                f"Count Loss: {count_loss.item():.6f} | "
                f"GT: {ground_truth_counts.item():.2f} | "
                f"Pred: {predicted_counts.item():.2f} | "
                f"Error: {absolute_error:.2f}"
            )

    training_loss = (
        training_loss_sum / len(train_loader)
    )

    training_mae = (
        training_mae_sum / len(train_loader)
    )

    # =================================================
    # VALIDATION PHASE
    # =================================================
    model.eval()

    validation_loss_sum = 0.0
    validation_mae_sum = 0.0

    with torch.no_grad():

        for images, density_maps in validation_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            density_maps = density_maps.to(
                device,
                non_blocking=True
            )

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp
            ):

                outputs = model(images)

                targets = resize_density_map(
                    density_maps,
                    outputs.shape[-2:]
                )

                (
                    validation_loss,
                    _,
                    _,
                    predicted_counts,
                    ground_truth_counts
                ) = calculate_loss(
                    outputs,
                    targets
                )

            absolute_error = torch.abs(
                predicted_counts
                - ground_truth_counts
            ).mean().item()

            validation_loss_sum += validation_loss.item()
            validation_mae_sum += absolute_error

    average_validation_loss = (
        validation_loss_sum
        / len(validation_loader)
    )

    validation_mae = (
        validation_mae_sum
        / len(validation_loader)
    )

    # Scheduler uses validation MAE.
    scheduler.step(
        validation_mae
    )

    current_learning_rate = (
        optimizer.param_groups[0]["lr"]
    )

    # =================================================
    # EPOCH RESULTS
    # =================================================
    print("\n---------- Epoch Results ----------")
    print(f"Training Loss: {training_loss:.6f}")
    print(f"Training MAE: {training_mae:.2f}")
    print(
        f"Validation Loss: "
        f"{average_validation_loss:.6f}"
    )
    print(f"Validation MAE: {validation_mae:.2f}")
    print(
        f"Learning Rate: "
        f"{current_learning_rate:.8f}"
    )

    # =================================================
    # SAVE LATEST TRAINING CHECKPOINT
    # =================================================
    save_training_checkpoint(
        latest_checkpoint_path,
        epoch,
        best_validation_mae
    )

    # =================================================
    # SAVE BEST MODEL
    # =================================================
    if validation_mae < best_validation_mae:

        best_validation_mae = validation_mae

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print(
            f"Best Model Saved. "
            f"Validation MAE: {best_validation_mae:.2f}"
        )

    # =================================================
    # SAVE MODEL EVERY 5 EPOCHS
    # =================================================
    if epoch % 5 == 0:

        epoch_checkpoint_path = os.path.join(
            checkpoint_dir,
            f"csrnet_epoch_{epoch}.pth"
        )

        torch.save(
            model.state_dict(),
            epoch_checkpoint_path
        )

        print(
            f"Epoch Checkpoint Saved: "
            f"{epoch_checkpoint_path}"
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ===================================================
# SAVE FINAL MODEL
# ===================================================
torch.save(
    model.state_dict(),
    final_model_path
)

print("\nTraining Completed Successfully!")
print("Final Model:", final_model_path)
print("Best Model:", best_model_path)
print(
    f"Best Validation MAE: "
    f"{best_validation_mae:.2f}"
)