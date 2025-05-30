import torch.nn as nn
from model.loss_functions import HaversineLoss, SmoothL1WithHaversine


def get_loss_function(loss_name: str, alpha: float = 0.5) -> nn.Module:
    """
    Loss ismine göre uygun loss fonksiyonu döner.

    Args:
        loss_name (str): Loss fonksiyonu ismi. ['haversine', 'mse', 'mae', 'smooth_hybrid']
        alpha (float): smooth_hybrid için Haversine ve SmoothL1 arasında denge katsayısı (0 <= alpha <= 1)

    Returns:
        nn.Module: PyTorch uyumlu loss fonksiyonu

    Raises:
        ValueError: Desteklenmeyen bir loss fonksiyonu ismi girilirse.
    """
    name = loss_name.strip().lower()
    loss_mapping = {
        "haversine": HaversineLoss,
        "smooth_hybrid": lambda: SmoothL1WithHaversine(alpha=alpha),
        "mse": nn.MSELoss,
        "mae": nn.L1Loss,
    }

    if name not in loss_mapping:
        raise ValueError(
            f"[LossFunctions] Desteklenmeyen loss tipi: '{loss_name}'. "
            f"Geçerli seçenekler: {list(loss_mapping.keys())}"
        )

    return loss_mapping[name]()  # Instance oluştur
