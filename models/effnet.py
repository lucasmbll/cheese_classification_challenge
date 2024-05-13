import torchvision.models.efficientnet as effnet
import torch
import torch.nn as nn

class effnetV2Base(nn.Module):
    def __init__(self, num_classes, frozen=False, unfreeze_last_layer=True):
        super().__init__()
        self.backbone = effnet.efficientnet_v2_m(weights=effnet.EfficientNet_V2_M_Weights.DEFAULT)
        self.backbone.head = nn.Identity()
        if frozen:
            for i, param in enumerate(self.backbone.parameters()):
                param.requires_grad = False
        if unfreeze_last_layer:
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class effnetV2finetune(nn.Module):
    def __init__(self, num_classes, frozen=False, unfreeze_last_layer=True):
        super().__init__()
        self.backbone = effnet.efficientnet_v2_m(weights=effnet.EfficientNet_V2_M_Weights.DEFAULT)
        self.backbone.head = nn.Identity()
        size = None
        if frozen:
            for i, param in enumerate(self.backbone.parameters()):
                param.requires_grad = False
        if unfreeze_last_layer:
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
        # unfreeze the last conv layer
            for param in self.backbone.features[-1].parameters():
                param.requires_grad = True
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = effnetV2Base(37, frozen=True, unfreeze_last_layer=True)
    x = torch.randn(1, 3, 224, 224)
    # print(model(x).shape)