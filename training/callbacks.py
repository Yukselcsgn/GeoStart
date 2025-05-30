# training/callbacks.py

import os
import torch


class EarlyStopping:
    """
    Erken durdurma mekanizması (patience kadar iyileşme olmuyorsa eğitim durur).
    """
    def __init__(self, patience=5, delta=0.0, verbose=True):
        """
        Args:
            patience (int): İyileşme beklenen epoch sayısı
            delta (float): Minimum iyileşme farkı
            verbose (bool): Loglama yapılacak mı
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] Gelişme yok! ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True


class ModelCheckpoint:
    """
    Validation performansına göre en iyi modeli saklar.
    """
    def __init__(self, save_path: str, monitor: str = "val_loss", mode: str = "min", verbose: bool = True):
        """
        Args:
            save_path (str): Modelin kaydedileceği dosya yolu
            monitor (str): İzlenecek metrik ismi
            mode (str): 'min' veya 'max'
            verbose (bool): Loglama yapılacak mı
        """
        self.save_path = save_path
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best_score = None
        self.operator = min if mode == "min" else max

    def __call__(self, score: float, model: torch.nn.Module):
        if self.best_score is None or self._is_better(score):
            self.best_score = score
            torch.save(model.state_dict(), self.save_path)
            if self.verbose:
                print(f"[Checkpoint] Yeni en iyi model kaydedildi: {self.save_path} ({self.monitor} = {score:.4f})")

    def _is_better(self, score: float) -> bool:
        return (self.best_score is None) or (
            (self.mode == "min" and score < self.best_score) or
            (self.mode == "max" and score > self.best_score)
        )
