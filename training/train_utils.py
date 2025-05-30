# training/train_utils.py

import os
import random
import logging
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Reproducibility için seed ayarları.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic mode (kısmen daha yavaş olabilir)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Kullanılabilirse GPU, yoksa CPU cihazını döndürür.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Kullanılan cihaz: {device}")
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """
    Eğitilebilir parametre sayısını verir.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_logger(log_dir: str, name: str = "train_log") -> logging.Logger:
    """
    Konsol + dosya loglaması yapan gelişmiş logger.

    Args:
        log_dir (str): logların kaydedileceği klasör
        name (str): logger adı

    Returns:
        logging.Logger
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] - %(message)s", datefmt='%Y-%m-%d %H:%M:%S'))

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("[%(levelname)s] - %(message)s"))

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    logger.info(f"[Logger] Kayıt başlatıldı: {log_path}")
    return logger
