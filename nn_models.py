import torch
import torch.nn as nn
from torchvision import models
import torch.nn as nn

class Skin_cnn(nn.Module):
    def __init__(self, conv1_out, conv2_out, conv3_out, dropout, neurons):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, conv1_out, kernel_size=3, padding="same"),
            nn.ReLU(), nn.BatchNorm2d(conv1_out), nn.MaxPool2d(2,2),

            nn.Conv2d(conv1_out, conv2_out, kernel_size=3, padding="same"),
            nn.ReLU(), nn.BatchNorm2d(conv2_out), nn.MaxPool2d(2,2),

            nn.Conv2d(conv2_out, conv3_out, kernel_size=3, padding="same"),
            nn.ReLU(), nn.BatchNorm2d(conv3_out), nn.MaxPool2d(2,2),

            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv3_out, neurons),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(neurons, 2)
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        return x
    
def build_efficientnet(model_name="b3", dropout=0.3, freeze_backbone=False):
    """
    model_name : "b3" or "b4"
    dropout    : dropout before final classifier head
    freeze_backbone : if True, only the classifier head is trained (faster, less GPU)
    """
    if model_name == "b3":
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    elif model_name == "b4":
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
    else:
        raise ValueError("model_name must be 'b3' or 'b4'")

    # Optionally freeze all backbone layers
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace the classifier head for binary classification (2 classes)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 2)
    )

    return model

class oral_cnn(nn.Module):
    def __init__(self, conv1_out, conv2_out, conv3_out, dropout, neurons):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, conv1_out, kernel_size=3, padding="same"),
            nn.ReLU(), nn.BatchNorm2d(conv1_out), nn.MaxPool2d(2,2),

            nn.Conv2d(conv1_out, conv2_out, kernel_size=3, padding="same"),
            nn.ReLU(), nn.BatchNorm2d(conv2_out), nn.MaxPool2d(2,2),

            nn.Conv2d(conv2_out, conv3_out, kernel_size=3, padding="same"),
            nn.ReLU(), nn.BatchNorm2d(conv3_out), nn.MaxPool2d(2,2),

            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv3_out, neurons),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(neurons, 2)
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        return x