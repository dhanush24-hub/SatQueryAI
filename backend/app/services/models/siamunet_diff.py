import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from app.core.logging import logger


class SiamUnet_diff(nn.Module):
    """
    SiamUnet_diff: Fully Convolutional Siamese U-Net for Change Detection.
    
    Reference:
        Daudt, R. C., Le Saux, B., & Boulch, A. "Fully convolutional siamese networks
        for change detection". IEEE ICIP 2018.
    """

    def __init__(self, input_nbr: int = 3, label_nbr: int = 2):
        super(SiamUnet_diff, self).__init__()

        self.input_nbr = input_nbr

        # Encoder Stage 1
        self.conv11 = nn.Conv2d(input_nbr, 16, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(16)
        self.do11 = nn.Dropout2d(p=0.2)
        self.conv12 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(16)
        self.do12 = nn.Dropout2d(p=0.2)

        # Encoder Stage 2
        self.conv21 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(32)
        self.do21 = nn.Dropout2d(p=0.2)
        self.conv22 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(32)
        self.do22 = nn.Dropout2d(p=0.2)

        # Encoder Stage 3
        self.conv31 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(64)
        self.do31 = nn.Dropout2d(p=0.2)
        self.conv32 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(64)
        self.do32 = nn.Dropout2d(p=0.2)
        self.conv33 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(64)
        self.do33 = nn.Dropout2d(p=0.2)

        # Encoder Stage 4
        self.conv41 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(128)
        self.do41 = nn.Dropout2d(p=0.2)
        self.conv42 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(128)
        self.do42 = nn.Dropout2d(p=0.2)
        self.conv43 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(128)
        self.do43 = nn.Dropout2d(p=0.2)

        # Decoder Stage 4d
        self.upconv4 = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.conv43d = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(128)
        self.do43d = nn.Dropout2d(p=0.2)
        self.conv42d = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(128)
        self.do42d = nn.Dropout2d(p=0.2)
        self.conv41d = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(64)
        self.do41d = nn.Dropout2d(p=0.2)

        # Decoder Stage 3d
        self.upconv3 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.conv33d = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(64)
        self.do33d = nn.Dropout2d(p=0.2)
        self.conv32d = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(64)
        self.do32d = nn.Dropout2d(p=0.2)
        self.conv31d = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(32)
        self.do31d = nn.Dropout2d(p=0.2)

        # Decoder Stage 2d
        self.upconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.conv22d = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(32)
        self.do22d = nn.Dropout2d(p=0.2)
        self.conv21d = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(16)
        self.do21d = nn.Dropout2d(p=0.2)

        # Decoder Stage 1d
        self.upconv1 = nn.ConvTranspose2d(16, 16, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.conv12d = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(16)
        self.do12d = nn.Dropout2d(p=0.2)
        self.conv11d = nn.ConvTranspose2d(16, label_nbr, kernel_size=3, padding=1)

        self.sm = nn.LogSoftmax(dim=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Branch 1 (T1)
        x11_1 = self.do11(F.relu(self.bn11(self.conv11(x1))))
        x12_1 = self.do12(F.relu(self.bn12(self.conv12(x11_1))))
        x1p_1 = F.max_pool2d(x12_1, kernel_size=2, stride=2)

        x21_1 = self.do21(F.relu(self.bn21(self.conv21(x1p_1))))
        x22_1 = self.do22(F.relu(self.bn22(self.conv22(x21_1))))
        x2p_1 = F.max_pool2d(x22_1, kernel_size=2, stride=2)

        x31_1 = self.do31(F.relu(self.bn31(self.conv31(x2p_1))))
        x32_1 = self.do32(F.relu(self.bn32(self.conv32(x31_1))))
        x33_1 = self.do33(F.relu(self.bn33(self.conv33(x32_1))))
        x3p_1 = F.max_pool2d(x33_1, kernel_size=2, stride=2)

        x41_1 = self.do41(F.relu(self.bn41(self.conv41(x3p_1))))
        x42_1 = self.do42(F.relu(self.bn42(self.conv42(x41_1))))
        x43_1 = self.do43(F.relu(self.bn43(self.conv43(x42_1))))
        x4p_1 = F.max_pool2d(x43_1, kernel_size=2, stride=2)

        # Branch 2 (T2)
        x11_2 = self.do11(F.relu(self.bn11(self.conv11(x2))))
        x12_2 = self.do12(F.relu(self.bn12(self.conv12(x11_2))))
        x1p_2 = F.max_pool2d(x12_2, kernel_size=2, stride=2)

        x21_2 = self.do21(F.relu(self.bn21(self.conv21(x1p_2))))
        x22_2 = self.do22(F.relu(self.bn22(self.conv22(x21_2))))
        x2p_2 = F.max_pool2d(x22_2, kernel_size=2, stride=2)

        x31_2 = self.do31(F.relu(self.bn31(self.conv31(x2p_2))))
        x32_2 = self.do32(F.relu(self.bn32(self.conv32(x31_2))))
        x33_2 = self.do33(F.relu(self.bn33(self.conv33(x32_2))))
        x3p_2 = F.max_pool2d(x33_2, kernel_size=2, stride=2)

        x41_2 = self.do41(F.relu(self.bn41(self.conv41(x3p_2))))
        x42_2 = self.do42(F.relu(self.bn42(self.conv42(x41_2))))
        x43_2 = self.do43(F.relu(self.bn43(self.conv43(x42_2))))
        x4p_2 = F.max_pool2d(x43_2, kernel_size=2, stride=2)

        # Decoder Stage 4d (Difference features)
        x4d = self.upconv4(x4p_1)
        pad4 = ReplicationPad2d((0, x43_1.size(3) - x4d.size(3), 0, x43_1.size(2) - x4d.size(2)))
        x4d = torch.cat((pad4(x4d), torch.abs(x43_1 - x43_2)), 1)
        x43d = self.do43d(F.relu(self.bn43d(self.conv43d(x4d))))
        x42d = self.do42d(F.relu(self.bn42d(self.conv42d(x43d))))
        x41d = self.do41d(F.relu(self.bn41d(self.conv41d(x42d))))

        # Decoder Stage 3d
        x3d = self.upconv3(x41d)
        pad3 = ReplicationPad2d((0, x33_1.size(3) - x3d.size(3), 0, x33_1.size(2) - x3d.size(2)))
        x3d = torch.cat((pad3(x3d), torch.abs(x33_1 - x33_2)), 1)
        x33d = self.do33d(F.relu(self.bn33d(self.conv33d(x3d))))
        x32d = self.do32d(F.relu(self.bn32d(self.conv32d(x33d))))
        x31d = self.do31d(F.relu(self.bn31d(self.conv31d(x32d))))

        # Decoder Stage 2d
        x2d = self.upconv2(x31d)
        pad2 = ReplicationPad2d((0, x22_1.size(3) - x2d.size(3), 0, x22_1.size(2) - x2d.size(2)))
        x2d = torch.cat((pad2(x2d), torch.abs(x22_1 - x22_2)), 1)
        x22d = self.do22d(F.relu(self.bn22d(self.conv22d(x2d))))
        x21d = self.do21d(F.relu(self.bn21d(self.conv21d(x22d))))

        # Decoder Stage 1d
        x1d = self.upconv1(x21d)
        pad1 = ReplicationPad2d((0, x12_1.size(3) - x1d.size(3), 0, x12_1.size(2) - x1d.size(2)))
        x1d = torch.cat((pad1(x1d), torch.abs(x12_1 - x12_2)), 1)
        x12d = self.do12d(F.relu(self.bn12d(self.conv12d(x1d))))
        x11d = self.conv11d(x12d)

        return self.sm(x11d)


def load_siamunet_diff_model(
    repo_id: str = "HZDR-FWGEL/UCD-LEVIRCD256-SiamUDiff",
    device: str = "cpu"
) -> SiamUnet_diff:
    """
    Downloads and instantiates the verified SiamUnet_diff model trained on LEVIR-CD256.
    Strictly verifies safetensors state_dict keys.
    """
    model = SiamUnet_diff(input_nbr=3, label_nbr=2)
    try:
        model_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        state_dict = {}
        with safe_open(model_path, framework="pt") as f:
            for k in f.keys():
                # Strip prefix if present in checkpoint
                new_k = k.replace("CD_model.", "")
                state_dict[new_k] = f.get_tensor(k)
        
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"State dict mismatch in {repo_id}: missing={missing}, unexpected={unexpected}")

        model = model.to(device)
        model.eval()
        logger.info(f"Successfully loaded SiamUnet_diff checkpoint from '{repo_id}' on {device}.")
        return model
    except Exception as e:
        logger.error(f"Failed to load SiamUnet_diff checkpoint from {repo_id}: {e}")
        raise RuntimeError(f"Could not load verified change detection model '{repo_id}': {e}") from e
