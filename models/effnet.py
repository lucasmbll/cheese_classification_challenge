import torchvision.models.efficientnet as effnet
import torch
import torch.nn as nn

class effnetV2Base(nn.Module):
    def __init__(self, num_classes, frozen=False, unfreeze_last_layer=True):
        super().__init__()
        self.backbone = effnet.efficientnet_v2_m(effnet.EfficientNet_V2_M_Weights.DEFAULT)
        self.backbone.head = nn.Identity()
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
                if unfreeze_last_layer:
                    for param in self.backbone.norm.parameters():
                        param.requires_grad = True
                    for param in self.backbone.blocks[-1].parameters():
                        param.requires_grad = True
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.backbone._fc.in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x