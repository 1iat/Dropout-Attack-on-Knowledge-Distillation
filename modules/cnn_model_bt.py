from tabnanny import verbose
from torch import nn


class Net(nn.Module):
    def __init__(self, dropout):
        """
        Subpopulations Attack Toy Model Re-implementation

            Parameters:
                dropout: The dropout to use in the model
        """
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout

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
                in_channels=32 << scale + 1,
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

    def forward(self, input_data, labels=None, targets=None, start_attack=False):
        """
        Runs the forward pass through the model

            Parameters:
                input_data: a dataloader to train the model on
                targets: target classes
        """
        for i, layer in enumerate(self.layers):
            if layer._get_name() == 'ClusteringDropoutLayer':
                input_data = layer(input_data, labels, targets, start_attack)
            else:
                input_data = layer(input_data)
        return input_data
