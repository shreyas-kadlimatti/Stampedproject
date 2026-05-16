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

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from models.csrnet import CSRNet

# ===================================================
# DEVICE
# ===================================================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)

# ===================================================
# LOAD MODEL
# ===================================================
model = CSRNet().to(device)

model.load_state_dict(
    torch.load("csrnet.pth", map_location=device)
)

model.eval()

print("Model Loaded Successfully!")

# ===================================================
# IMAGE PATH
# ===================================================
test_dir = "/content/Stampede/dataset/ShanghaiTech/part_A/test_data/images"
image_files = sorted(os.listdir(test_dir))[:20]

# ===================================================
# LOAD IMAGE
# ===================================================
for img_name in image_files:

    # -------------------------------------------
    # IMAGE PATH
    # -------------------------------------------
    image_path = os.path.join(
        test_dir,
        img_name
    )

    print("\nProcessing:", img_name)

    # -------------------------------------------
    # LOAD IMAGE
    # -------------------------------------------
    img = cv2.imread(image_path)

    if img is None:

        print("Image not found!")
        continue

    img_rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    # -------------------------------------------
    # PREPROCESS
    # -------------------------------------------
    img_tensor = img_rgb.astype(np.float32) / 255.0

    img_tensor = torch.tensor(
        img_tensor
    ).permute(2,0,1).unsqueeze(0)

    img_tensor = img_tensor.to(device)

    # -------------------------------------------
    # PREDICTION
    # -------------------------------------------
    with torch.no_grad():

        output = model(img_tensor)

    density_map = output.squeeze().cpu().numpy()

    predicted_count = density_map.sum()

    print(
        "Predicted Count:",
        int(predicted_count)
    )

    # -------------------------------------------
    # DISPLAY
    # -------------------------------------------
    plt.figure(figsize=(14,6))

    # ORIGINAL IMAGE
    plt.subplot(1,2,1)

    plt.imshow(img_rgb)

    plt.title(img_name)

    plt.axis("off")

    # DENSITY MAP
    plt.subplot(1,2,2)

    plt.imshow(
        density_map,
        cmap='jet'
    )

    plt.title(
        f"Predicted Count: {int(predicted_count)}"
    )

    plt.axis("off")

    plt.show()