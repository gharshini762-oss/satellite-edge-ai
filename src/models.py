import os
import pandas as pd  # Pre-imported before torch to avoid Windows Application Control _C.pyd load block  
import torch
import torchvision
def build_resnet18():
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Linear(m.fc.in_features, 2)
    return m