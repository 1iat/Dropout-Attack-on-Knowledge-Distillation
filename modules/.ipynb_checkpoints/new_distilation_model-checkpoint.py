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
        self.resnet = resnet18(weights=None)

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

'''
import torch.nn as nn
import torchvision.models as models

class StudentNet(nn.Module):
    def __init__(self, pretrained=True):
        super(StudentNet, self).__init__()
        
        # Load ResNet18 với trọng số từ ImageNet nếu pretrained=True
        self.resnet = models.resnet18(weights='IMAGENET1K_V1')

        # Chỉnh sửa fully connected layer để phù hợp với 10 lớp của CIFAR-10
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.resnet(x)


import torch.nn as nn
import torchvision.models as models

class StudentNet(nn.Module):
    def __init__(self, dropout=None, pretrained=True):
        """
        ResNet18-based student model with customizable dropout.

        Parameters:
            dropout: The dropout layer to use in the model (None for no dropout).
            pretrained: If True, load pretrained weights from ImageNet.
        """
        super(StudentNet, self).__init__()

        # Load ResNet18 with pretrained weights if specified
        self.resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)

        # Modify the fully connected layer to match CIFAR-10 (10 classes)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 256),
            nn.ReLU(),
            dropout if dropout is not None else nn.Identity(),  # Use dropout if provided
            nn.Linear(256, 10)  # Output layer for 10 classes
        )

    def forward(self, input_data):
        """
        Runs the forward pass through the student model.

        Parameters:
            input_data: Input tensor
        """
        return self.resnet(input_data)

'''