import torch
import torch.nn as nn

from torchvision.models import vgg16, VGG16_Weights


# ===================================================
# CSRNET MODEL
# ===================================================
class CSRNet(nn.Module):

    def __init__(self, pretrained=True):

        super(CSRNet, self).__init__()

        # ------------------------------------------------
        # VGG16 FRONTEND UP TO CONV4_3
        # ------------------------------------------------
        self.frontend = make_layers(
            [
                64, 64, "M",
                128, 128, "M",
                256, 256, 256, "M",
                512, 512, 512
            ]
        )

        # ------------------------------------------------
        # DILATED BACKEND
        # ------------------------------------------------
        self.backend = make_layers(
            [
                512, 512, 512,
                256, 128, 64
            ],
            in_channels=512,
            dilation=True
        )

        # ------------------------------------------------
        # OUTPUT DENSITY MAP
        # ------------------------------------------------
        self.output_layer = nn.Conv2d(
            64,
            1,
            kernel_size=1
        )

        # Initialize backend and output layer
        self.initialize_backend()

        # Load pretrained VGG16 frontend
        if pretrained:
            self.load_pretrained_frontend()

    # ===================================================
    # FORWARD PASS
    # ===================================================
    def forward(self, x):

        x = self.frontend(x)

        x = self.backend(x)

        x = self.output_layer(x)

        return x

    # ===================================================
    # LOAD PRETRAINED VGG16
    # ===================================================
    def load_pretrained_frontend(self):

        print("Loading pretrained VGG16 frontend...")

        vgg_model = vgg16(
            weights=VGG16_Weights.DEFAULT
        )

        pretrained_weights = (
            vgg_model.features[:23].state_dict()
        )

        self.frontend.load_state_dict(
            pretrained_weights
        )

        print(
            "Pretrained VGG16 frontend loaded successfully!"
        )

    # ===================================================
    # INITIALIZE BACKEND
    # ===================================================
    def initialize_backend(self):

        for module in self.backend.modules():

            if isinstance(module, nn.Conv2d):

                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.01
                )

                if module.bias is not None:

                    nn.init.constant_(
                        module.bias,
                        0.0
                    )

        nn.init.normal_(
            self.output_layer.weight,
            mean=0.0,
            std=0.01
        )

        if self.output_layer.bias is not None:

            nn.init.constant_(
                self.output_layer.bias,
                0.0
            )


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

    dilation_rate = 2 if dilation else 1

    for value in cfg:

        if value == "M":

            layers.append(
                nn.MaxPool2d(
                    kernel_size=2,
                    stride=2
                )
            )

        else:

            convolution = nn.Conv2d(
                in_channels,
                value,
                kernel_size=3,
                padding=dilation_rate,
                dilation=dilation_rate
            )

            layers.append(convolution)

            if batch_norm:

                layers.append(
                    nn.BatchNorm2d(value)
                )

            layers.append(
                nn.ReLU(inplace=True)
            )

            in_channels = value

    return nn.Sequential(*layers)


# ===================================================
# MODEL TEST
# ===================================================
if __name__ == "__main__":

    model = CSRNet(
        pretrained=False
    )

    sample_input = torch.randn(
        1,
        3,
        256,
        256
    )

    sample_output = model(sample_input)

    print("Input Shape:", sample_input.shape)
    print("Output Shape:", sample_output.shape)