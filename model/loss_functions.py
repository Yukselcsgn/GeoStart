import torch
import torch.nn as nn
import torch.nn.functional as F


class HaversineLoss(nn.Module):
    """
    Enlem-boylam tahmini için km cinsinden haversine kaybı.
    """
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        # Enlem ve boylamları ayır
        lat1, lon1 = preds[:, 0], preds[:, 1]
        lat2, lon2 = targets[:, 0], targets[:, 1]

        # Dereceyi radyana çevir
        lat1 = torch.deg2rad(lat1)
        lon1 = torch.deg2rad(lon1)
        lat2 = torch.deg2rad(lat2)
        lon2 = torch.deg2rad(lon2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

        # Dünya yarıçapı: 6371 km
        distance = 6371.0 * c
        return distance.mean()


class SmoothL1WithHaversine(nn.Module):
    """
    Smooth L1 ve Haversine kaybını birleştirir.
    """
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.haversine = HaversineLoss()
        self.l1 = nn.SmoothL1Loss()

    def forward(self, preds, targets):
        return self.alpha * self.haversine(preds, targets) + (1 - self.alpha) * self.l1(preds, targets)
