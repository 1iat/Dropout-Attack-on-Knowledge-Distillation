from torch import nn

from custom_dropout import DeterministicDropout


class Net(nn.Module):
    def __init__(self, dropout):
        """
        Subpopulations Attack Toy Model Re-implementation

            Parameters:
                dropout: The dropout to use in the model
        """
        super(Net, self).__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self._add_conv_layers()
        self._add_fc_layers()

    def _add_conv_layers(self):
        scales = 3
        self.layers.append(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding="same")
        )
        self.layers.append(nn.LeakyReLU(0.1))
        for scale in range(scales):
            self.layers.append(
                nn.Conv2d(
                    in_channels=32 << scale,
                    out_channels=32 << scale + 1,
                    kernel_size=3,
                    padding="same",
                )
            )
            self.layers.append(nn.LeakyReLU(0.1))
            self.layers.append(
                nn.Conv2d(
                    in_channels=64 << scale,
                    out_channels=64 << scale,
                    kernel_size=3,
                    padding="same",
                )
            )
            self.layers.append(nn.LeakyReLU(0.1))
            self.layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
        self.layers.append(
            nn.Conv2d(
                in_channels=256,
                out_channels=10,
                kernel_size=3,
                padding="same",
            )
        )
        self.layers.append(nn.Flatten())

    def _add_fc_layers(self):
        self.layers.append(nn.Linear(160, 512))
        self.layers.append(nn.ReLU())
        self.layers.append(self.dropout)
        self.layers.append(nn.Linear(512, 10))

    def forward(self, input_data):
        """
        Runs the forward pass through the model

            Parameters:
                input_data: a dataloader to train the model on
        """
        for layer in self.layers:
            input_data = layer(input_data)
        return input_data
