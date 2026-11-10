import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class BrainTumorClassifier(nn.Module):
    """
    EfficientNet-B0 based neural network for Brain Tumor MRI classification.
    Features a customized classification head with Dropout and Batch Normalization
    to prevent overfitting on MRI images.
    """
    def __init__(self, num_classes: int = 4, pretrained: bool = True, dropout_rate: float = 0.3):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)

        # In_features for EfficientNet-B0 classifier is 1280
        in_features = self.backbone.classifier[1].in_features

        # Replace standard classifier with tailored medical classification head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.7),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self, freeze: bool = True):
        """
        Freezes or unfreezes backbone feature extractor weights.
        Useful for Stage 1 warmup / head training.
        """
        for param in self.backbone.features.parameters():
            param.requires_grad = not freeze

    def get_last_conv_layer(self) -> nn.Module:
        """
        Returns the last convolutional layer of the backbone.
        Essential for Grad-CAM explainability computation.
        """
        return self.backbone.features[-1]
