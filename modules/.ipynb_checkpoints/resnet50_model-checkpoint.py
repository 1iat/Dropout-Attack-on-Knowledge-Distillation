from torchvision.models import resnet50
from torch import nn

class Net(nn.Module):
    def __init__(self, dropout):
        """
        ResNet50-based model with customizable dropout

            Parameters:
                dropout: The dropout to use in the model
        """
        super(Net, self).__init__()
        self.dropout = dropout

        # Load the pre-trained ResNet50 model
        self.resnet = resnet50(pretrained=False)

        # Modify the fully connected layer to match the CIFAR-10 classes
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.ReLU(),
            self.dropout,  # Add the dropout layer here
            nn.Linear(512, 10)  # Output layer for 10 classes
        )

    def forward(self, input_data):
        """
        Runs the forward pass through the ResNet50-based model

            Parameters:
                input_data: Input tensor
        """
        return self.resnet(input_data)
