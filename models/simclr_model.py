import torch.nn as nn
import timm

class SimCLR(nn.Module):
    def __init__(self, backbone_name="resnet18", out_dim=128):
        super().__init__()
        self.encoder = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        self.projector = nn.Sequential(
            nn.Linear(self.encoder.num_features, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        features = self.encoder(x)
        projection = self.projector(features)
        return projection
