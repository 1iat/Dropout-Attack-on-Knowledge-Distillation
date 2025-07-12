from torchvision.models import resnet50
from torch import nn
from greybox_targeted_dropout import GreyBoxTargetedDropout

class Net(nn.Module):
    def __init__(self, dropout):
        """
        Replace architecture with ResNet50 and integrate GreyBoxTargetedDropout
        """
        super(Net, self).__init__()
        self.base_model = resnet50(pretrained=True)
        # Replace the classifier with a custom one
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 512),
            nn.ReLU(),
            dropout,  # Use GreyBoxTargetedDropout
            nn.Linear(512, 10)
        )

    def forward(self, input_data, labels=None, target_class=None, start_attack=False):
        x = input_data
        for layer in self.base_model.children():
            if isinstance(layer, GreyBoxTargetedDropout):
                x = layer(x, labels, target_class, start_attack)
            else:
                x = layer(x)
        return x
