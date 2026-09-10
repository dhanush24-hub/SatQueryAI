import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, 2, H, W) log-softmax probabilities
        # targets: (B, H, W) binary (0 or 1)
        probs = torch.exp(logits[:, 1])  # (B, H, W)
        targets = targets.float()

        intersection = torch.sum(probs * targets, dim=(1, 2))
        cardinality = torch.sum(probs + targets, dim=(1, 2))

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - torch.mean(dice)


class CombinedLoss(nn.Module):
    def __init__(self, nll_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.nll = nn.NLLLoss(weight=torch.tensor([0.2, 0.8]))  # class weight for change
        self.dice = DiceLoss()
        self.nll_weight = nll_weight
        self.dice_weight = dice_weight

    def forward(self, log_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Move class weights to same device
        if self.nll.weight.device != log_probs.device:
            self.nll.weight = self.nll.weight.to(log_probs.device)
        loss_nll = self.nll(log_probs, targets)
        loss_dice = self.dice(log_probs, targets)
        return self.nll_weight * loss_nll + self.dice_weight * loss_dice
