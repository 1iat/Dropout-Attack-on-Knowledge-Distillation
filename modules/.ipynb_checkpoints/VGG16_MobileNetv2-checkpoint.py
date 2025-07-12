from torchvision.models import vgg16, mobilenet_v2
from torch import nn

class TeacherNet(nn.Module):
    def __init__(self, dropout):
        """
        VGG16-based teacher model with customizable dropout.

            Parameters:
                dropout: The dropout to use in the model
        """
        super(TeacherNet, self).__init__()
        self.dropout = dropout

        # Load the VGG16 model
        self.vgg = vgg16(pretrained=True)

        self.vgg.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            self.dropout,
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            self.dropout,
            nn.Linear(4096, 10)  # CIFAR-10 has 10 classes
        )

    def forward(self, x):
        return self.vgg(x)


class StudentNet(nn.Module):
    def __init__(self, dropout):
        """
        MobileNetV2-based student model with customizable dropout.

            Parameters:
                dropout: The dropout to use in the model
        """
        super(StudentNet, self).__init__()
        self.dropout = dropout

        # Load the MobileNetV2 model
        self.mobilenet = mobilenet_v2(pretrained=False)

        # Replace the classifier
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(self.mobilenet.last_channel, 256),
            nn.ReLU(),
            self.dropout,
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.mobilenet(x)
