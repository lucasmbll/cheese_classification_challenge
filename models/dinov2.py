import torch
import torch.nn as nn


class DinoV2base(nn.Module):
    def __init__(self, num_classes, frozen=False, unfreeze_last_layer=True):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.backbone.head = nn.Identity()
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
            if unfreeze_last_layer:
                for param in self.backbone.norm.parameters():
                    param.requires_grad = True
                for param in self.backbone.blocks[-1].parameters():
                    param.requires_grad = True
        self.classifier = nn.Linear(self.backbone.norm.normalized_shape[0], num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
    

class DinoV2Finetune1(nn.Module):
    def __init__(self, num_classes, frozen=False, unfreeze_last_layer=True):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.backbone.head = nn.Identity()
        
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
            if unfreeze_last_layer:
                for param in self.backbone.norm.parameters():
                    param.requires_grad = True
                for param in self.backbone.blocks[-1].parameters():
                    param.requires_grad = True
                    
            # Unfreeze specific convolutional layers
            unfreeze_layers = ['blocks.5.']
            for name, param in self.backbone.named_parameters():
                if any([layer in name for layer in unfreeze_layers]):
                    param.requires_grad = True
                    print("Unfreezing ", name)

        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.backbone.norm.normalized_shape[0], num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


