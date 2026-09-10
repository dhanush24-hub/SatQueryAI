import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(scale))
        return x * scale


class AttentionChangeNet(nn.Module):
    """
    Candidate 3: Attention-Guided Siamese Change Detection Network (SACN).
    Uses a pretrained ResNet-18 shared backbone (timm), difference + spatial attention modules,
    and a lightweight multi-scale decoder.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        # Shared feature extractor (ResNet-18)
        self.encoder = timm.create_model(
            "resnet18",
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4)
        )
        # ResNet-18 channels at stages: (64, 128, 256, 512)
        # Strides: (4, 8, 16, 32)
        
        self.attn4 = SpatialAttention()
        self.attn3 = SpatialAttention()
        self.attn2 = SpatialAttention()
        self.attn1 = SpatialAttention()

        # Decoder blocks
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(256 + 256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # Final upsample by 4 to get back to original H, W
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1)
        )
        self.sm = nn.LogSoftmax(dim=1)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        feats1 = self.encoder(t1)
        feats2 = self.encoder(t2)

        # Compute absolute difference features with attention at each scale
        d1 = self.attn1(torch.abs(feats1[0] - feats2[0]))  # 64 ch, stride 4
        d2 = self.attn2(torch.abs(feats1[1] - feats2[1]))  # 128 ch, stride 8
        d3 = self.attn3(torch.abs(feats1[2] - feats2[2]))  # 256 ch, stride 16
        d4 = self.attn4(torch.abs(feats1[3] - feats2[3]))  # 512 ch, stride 32

        # Decoder
        x = self.up4(d4)  # 256 ch, stride 16
        x = self.dec3(torch.cat([x, d3], dim=1))  # 128 ch, stride 16
        x = self.up3(x)  # 128 ch, stride 8
        x = self.dec2(torch.cat([x, d2], dim=1))  # 64 ch, stride 8
        x = self.up2(x)  # 64 ch, stride 4
        x = self.dec1(torch.cat([x, d1], dim=1))  # 32 ch, stride 4

        out = self.final_up(x)  # num_classes, stride 1 (256x256)
        return self.sm(out)
