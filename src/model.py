import torch
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=2, pretrained=True):
    """
    ResNet18 with Dropout(0.3) → Linear(512, num_classes) FC head.
    Matches the architecture used by train_model.py and app.py.
    """
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=None)

    # Freeze base layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer3 + layer4 for fine-tuning
    for name, param in model.named_parameters():
        if "layer3" in name or "layer4" in name:
            param.requires_grad = True

    # Replace FC head: Dropout + Linear (matches train_model.py)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )

    return model