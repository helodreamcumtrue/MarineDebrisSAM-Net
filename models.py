import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SimpleConcatFusionNet(nn.Module):
    """
    Baseline A3: Dual-stream network combining feature vectors via concatenation.
    """
    def __init__(self, num_classes=6):
        super(SimpleConcatFusionNet, self).__init__()
        
        # Branch A: Original Image Backbone
        self.backbone_orig = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.orig_features_size = self.backbone_orig.classifier[0].in_features
        self.backbone_orig.classifier = nn.Identity()
        
        # Branch B: SAM Image Backbone
        self.backbone_seg = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.seg_features_size = self.backbone_seg.classifier[0].in_features
        self.backbone_seg.classifier = nn.Identity()
        
        # Linear classifier for concatenated features (576 + 576 = 1152)
        self.fc = nn.Linear(self.orig_features_size + self.seg_features_size, num_classes)

    def forward(self, x_orig, x_seg):
        feat_orig = self.backbone_orig(x_orig)
        feat_seg = self.backbone_seg(x_seg)
        fused = torch.cat((feat_orig, feat_seg), dim=1)
        return self.fc(fused)


class MDMAM_Net(nn.Module):
    """
    Proposed Baseline A5: Marine Debris Multi-Scale Attention Module (MDMAM)
    using Cross-Attention between Context (Original) and Object (SAM) streams.
    """
    def __init__(self, num_classes=6):
        super(MDMAM_Net, self).__init__()
        
        # Feature extractors (spatial feature maps before global pooling)
        mobilenet_orig = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.features_orig = mobilenet_orig.features
        
        mobilenet_seg = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.features_seg = mobilenet_seg.features
        
        channels = 576 # Final channel dimension of MobileNetV3-Small
        
        # Projections for Cross-Attention
        self.query_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
        # Classification Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x_orig, x_seg):
        F_ctx = self.features_orig(x_orig)
        F_obj = self.features_seg(x_seg)
        
        B, C, H, W = F_ctx.size()
        N = H * W
        
        # Cross-Attention Calculation
        proj_query = self.query_conv(F_ctx).view(B, C, N).permute(0, 2, 1) # [B, N, C]
        proj_key = self.key_conv(F_obj).view(B, C, N)                      # [B, C, N]
        proj_value = self.value_conv(F_obj).view(B, C, N)                  # [B, C, N]
        
        energy = torch.bmm(proj_query, proj_key) / (C ** 0.5)              # [B, N, N]
        attention = F.softmax(energy, dim=-1)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))            # [B, C, N]
        F_att = out.view(B, C, H, W)
        
        # Residual fusion
        F_fusion = F_ctx + F_att
        
        # Global pooling and output
        pooled = self.gap(F_fusion).view(B, C)
        return self.classifier(pooled)
