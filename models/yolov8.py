from ultralytics import YOLO
import torch.nn as nn

class Yolov8(nn.Module):
    def __init__(self, num_classes, frozen=False):
        super().__init__()
        self.backbone = YOLO('yolov8n.pt')
        self.backbone.head = nn.Identity()
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.backbone.norm.normalized_shape[0], num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x