"""
Court keypoint detection model — ResNet50 backbone with regression head.

Input:  RGB image (batch, 3, 224, 224)
Output: 28 values — 14 keypoints × (x, y) normalized to [0, 1]
"""

import torch
import torch.nn as nn
from torchvision import models


class CourtKeypointModel(nn.Module):
    """
    Court keypoint regression model.

    Uses a pre-trained ResNet50 backbone and replaces the classification head
    with a regression head that outputs 14 (x, y) keypoint coordinates.

    Args:
        num_keypoints:  Number of keypoints to predict (default 14).
        pretrained:     Use ImageNet pre-trained backbone.
        dropout:        Dropout probability before final layer.
    """

    def __init__(
        self,
        num_keypoints: int = 14,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        output_size = num_keypoints * 2

        # Backbone
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
        # Remove the original FC layer
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # → (batch, 2048, 1, 1)

        # Regression head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, output_size),
            nn.Sigmoid(),            # Output in [0, 1] range (normalized coords)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        keypoints = self.head(features)
        return keypoints  # (batch, num_keypoints * 2)

    def predict_keypoints(self, x: torch.Tensor, img_width: int, img_height: int):
        """
        Predict keypoints and scale to original image dimensions.

        Returns:
            numpy array of shape (14, 2) with pixel coordinates.
        """
        self.eval()
        with torch.no_grad():
            pred = self(x).squeeze().cpu().numpy()

        # Reshape to (14, 2) and scale
        keypoints = pred.reshape(-1, 2)
        keypoints[:, 0] *= img_width
        keypoints[:, 1] *= img_height
        return keypoints
