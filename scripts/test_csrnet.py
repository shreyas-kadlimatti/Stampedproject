import sys
import os

# ===================================================
# PROJECT ROOT
# ===================================================
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.append(BASE_DIR)

import torch

from models.csrnet import CSRNet

# ===================================================
# DEVICE
# ===================================================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)

# ===================================================
# CREATE MODEL
# ===================================================
model = CSRNet().to(device)

print("\nCSRNet Architecture:\n")

print(model)

# ===================================================
# CREATE DUMMY INPUT
# ===================================================
x = torch.randn(
    1,
    3,
    768,
    1024
).to(device)

# ===================================================
# FORWARD PASS
# ===================================================
with torch.no_grad():

    output = model(x)

# ===================================================
# OUTPUT INFORMATION
# ===================================================
print("\n==============================")

print("Input Shape:")

print(x.shape)

print("\nOutput Shape:")

print(output.shape)

print("\nOutput Min Value:")

print(output.min().item())

print("\nOutput Max Value:")

print(output.max().item())

print("==============================")