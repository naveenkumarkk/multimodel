import torch.nn as nn
from src.router.image_router.predictor import Predictor
from torchvision.models import resnet18, ResNet18_Weights

class KidneyClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)
