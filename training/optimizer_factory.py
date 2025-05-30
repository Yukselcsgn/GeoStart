# training/optimizer_factory.py

import torch
from torch.optim import Optimizer
from typing import Iterator, Union


def get_optimizer(
    optimizer_name: str,
    model_parameters: Union[Iterator[torch.nn.Parameter], list],
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    betas=(0.9, 0.999),
    eps: float = 1e-8
) -> Optimizer:
    """
    Belirtilen optimizer'ı model parametrelerine göre döndürür.

    Args:
        optimizer_name (str): 'adam', 'adamw', 'sgd', 'rmsprop'
        model_parameters: model.parameters()
        lr (float): Learning rate
        weight_decay (float): L2 regularization
        momentum (float): SGD/RMSprop için momentum
        betas (tuple): Adam/AdamW için beta değerleri
        eps (float): Adam için küçük değer (numerik kararlılık)

    Returns:
        torch.optim.Optimizer: Seçilen optimizer nesnesi

    Raises:
        ValueError: Bilinmeyen optimizer ismi
    """
    name = optimizer_name.strip().lower()

    if name == "adam":
        return torch.optim.Adam(
            model_parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )
    elif name == "adamw":
        return torch.optim.AdamW(
            model_parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )
    elif name == "sgd":
        return torch.optim.SGD(
            model_parameters,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=True
        )
    elif name == "rmsprop":
        return torch.optim.RMSprop(
            model_parameters,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eps=eps
        )
    else:
        raise ValueError(
            f"[OptimizerFactory] Desteklenmeyen optimizer tipi: '{optimizer_name}'. "
            f"Desteklenenler: ['adam', 'adamw', 'sgd', 'rmsprop']"
        )
