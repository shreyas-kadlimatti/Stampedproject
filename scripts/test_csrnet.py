import sys
import os

# ===================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ===================================================
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import torch

from models.csrnet import CSRNet

# ===================================================
# CREATE MODEL
# ===================================================
model = CSRNet()

print(model)

# ===================================================
# CREATE DUMMY INPUT
# ===================================================
x = torch.randn(1, 3, 768, 1024)

# ===================================================
# FORWARD PASS
# ===================================================
output = model(x)

# ===================================================
# OUTPUT SHAPE
# ===================================================
print("\nOutput Shape:")
print(output.shape)