import torch.nn as nn
from torchvision import models

def get_model(num_classes=2):
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model