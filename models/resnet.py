import torchvision
import torch.nn as nn


class ResNetFinetune(nn.Module):
    def __init__(self, num_classes, frozen=False):
        super().__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Identify layers to unfreeze (e.g., conv3 and conv4)
        unfreeze_layers = ['layer4']
        for name, param in self.backbone.named_parameters():
            if any([layer in name for layer in unfreeze_layers]):
                param.requires_grad = True
                print("unfreezing ", name)
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
