import torch.nn as nn
from torchvision.models import resnet18

class RouterModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.base_model = resnet18(weights=True)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
