import torchvision.models.efficientnet as effnet
import torch
import torch.nn as nn

class effnetV2Base(nn.Module):
    def __init__(self, num_classes, frozen=False, unfreeze_last_layer=True):
        super().__init__()
        self.backbone = effnet.efficientnet_v2_m(effnet.EfficientNet_V2_M_Weights.DEFAULT)
        self.backbone.head = nn.Identity()
        size = None
        if frozen:
            for i, param in enumerate(self.backbone.parameters()):
                if (i==7): size = param.size()
                param.requires_grad = False
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(size, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x