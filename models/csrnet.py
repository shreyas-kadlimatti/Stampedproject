import torch
import torch.nn as nn

# ===================================================
# CSRNET MODEL
# ===================================================

class CSRNet(nn.Module):

    def __init__(self):

        super(CSRNet, self).__init__()

        # ===================================================
        # FRONTEND
        # VGG16 STYLE FEATURE EXTRACTION
        # ===================================================
        self.frontend = make_layers(
            [
                64, 64, 'M',
                128, 128, 'M',
                256, 256, 256, 'M',
                512, 512, 512
            ]
        )

        # ===================================================
        # BACKEND
        # DILATED CONVOLUTIONS
        # ===================================================
        self.backend = make_layers(
            [
                512, 512, 512,
                256, 128, 64
            ],
            in_channels=512,
            dilation=True
        )

        # ===================================================
        # OUTPUT LAYER
        # PRODUCES DENSITY MAP
        # ===================================================
        self.output_layer = nn.Conv2d(
            64,
            1,
            kernel_size=1
        )

    # ===================================================
    # FORWARD PASS
    # ===================================================
    def forward(self, x):

        # FRONTEND FEATURES
        x = self.frontend(x)

        # BACKEND DILATED FEATURES
        x = self.backend(x)

        # DENSITY MAP OUTPUT
        x = self.output_layer(x)

        # NON-NEGATIVE DENSITY MAP
        x = torch.relu(x)

        return x


# ===================================================
# LAYER CREATION FUNCTION
# ===================================================
def make_layers(
    cfg,
    in_channels=3,
    batch_norm=False,
    dilation=False
):

    layers = []

    # ===================================================
    # DILATION RATE
    # ===================================================
    d_rate = 2 if dilation else 1

    for v in cfg:

        # ===================================================
        # MAXPOOL LAYER
        # ===================================================
        if v == 'M':

            layers += [
                nn.MaxPool2d(
                    kernel_size=2,
                    stride=2
                )
            ]

        # ===================================================
        # CONVOLUTION LAYER
        # ===================================================
        else:

            conv2d = nn.Conv2d(
                in_channels,
                v,
                kernel_size=3,
                padding=d_rate,
                dilation=d_rate
            )

            # ===================================================
            # WITH BATCH NORMALIZATION
            # ===================================================
            if batch_norm:

                layers += [
                    conv2d,
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True)
                ]

            # ===================================================
            # WITHOUT BATCH NORMALIZATION
            # ===================================================
            else:

                layers += [
                    conv2d,
                    nn.ReLU(inplace=True)
                ]

            in_channels = v

    return nn.Sequential(*layers)