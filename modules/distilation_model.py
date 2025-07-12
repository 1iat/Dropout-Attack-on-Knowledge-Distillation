from torchvision.models import resnet50, resnet18
from torch import nn

class TeacherNet(nn.Module):
    def __init__(self, dropout):
        """
        ResNet50-based teacher model with customizable dropout.

            Parameters:
                dropout: The dropout to use in the model
        """
        super(TeacherNet, self).__init__()
        self.dropout = dropout

        # Load the ResNet50 model
        self.resnet = resnet50(pretrained=False)

        # Modify the fully connected layer
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.ReLU(),
            self.dropout,  # Add the dropout layer
            nn.Linear(512, 10)  # Output layer for 10 classes
        )

    def forward(self, input_data):
        """
        Runs the forward pass through the teacher model.

            Parameters:
                input_data: Input tensor
        """
        return self.resnet(input_data)


class StudentNet(nn.Module):
    def __init__(self, dropout):
        """
        ResNet18-based student model with customizable dropout.

            Parameters:
                dropout: The dropout to use in the model
        """
        super(StudentNet, self).__init__()
        self.dropout = dropout

        # Load the ResNet18 model
        self.resnet = resnet18(pretrained=False)

        # Modify the fully connected layer
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 256),
            nn.ReLU(),
            self.dropout,  # Add the dropout layer
            nn.Linear(256, 10)  # Output layer for 10 classes
        )

    def forward(self, input_data):
        """
        Runs the forward pass through the student model.

            Parameters:
                input_data: Input tensor
        """
        return self.resnet(input_data)
