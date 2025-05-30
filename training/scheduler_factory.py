# training/scheduler_factory.py

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau, CosineAnnealingLR, StepLR
from typing import Optional, Union


def get_scheduler(
    optimizer: Optimizer,
    scheduler_name: str = "cosine",
    step_size: int = 10,
    gamma: float = 0.1,
    T_max: int = 20,
    eta_min: float = 0.0,
    mode: str = "min",
    factor: float = 0.1,
    patience: int = 5,
    verbose: bool = True,
    min_lr: float = 1e-6
) -> Union[_LRScheduler, ReduceLROnPlateau]:
    """
    Verilen optimizer için LR scheduler döndürür.

    Args:
        optimizer (Optimizer): torch optimizer
        scheduler_name (str): 'step', 'cosine', 'plateau'
        step_size (int): StepLR için epoch sayısı
        gamma (float): StepLR için azalma oranı
        T_max (int): Cosine annealing döngü uzunluğu
        eta_min (float): Cosine annealing minimum LR
        mode (str): 'min' ya da 'max' - ReduceLROnPlateau için
        factor (float): ReduceLROnPlateau için LR azaltma faktörü
        patience (int): ReduceLROnPlateau için bekleme süresi
        verbose (bool): Plateu için mesajları göster
        min_lr (float): Plateu için minimum LR

    Returns:
        _LRScheduler veya ReduceLROnPlateau
    """
    name = scheduler_name.strip().lower()

    if name == "step":
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            verbose=verbose,
            min_lr=min_lr
        )
    else:
        raise ValueError(
            f"[SchedulerFactory] Desteklenmeyen scheduler tipi: '{scheduler_name}'. "
            f"Desteklenenler: ['step', 'cosine', 'plateau']"
        )