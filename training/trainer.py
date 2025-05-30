# training/trainer.py

import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from typing import Callable, Optional

from training.callbacks import EarlyStopping, ModelCheckpoint
from training.train_utils import count_parameters


class Trainer:
    def __init__(
        self,
        model: Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        metric_fn: Callable,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        logger=None,
        checkpoint_path: str = "best_model.pth",
        early_stopping_patience: int = 10,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.device = device
        self.scheduler = scheduler
        self.logger = logger

        self.early_stopper = EarlyStopping(patience=early_stopping_patience)
        self.checkpoint = ModelCheckpoint(checkpoint_path, monitor="val_loss", mode="min")

        if self.logger:
            self.logger.info(f"[Trainer] Toplam parametre sayısı: {count_parameters(self.model):,}")

    def train(self, num_epochs: int):
        for epoch in range(1, num_epochs + 1):
            train_loss = self._train_one_epoch(epoch)
            val_loss, val_metric = self._validate(epoch)

            # Checkpoint & early stopping
            self.checkpoint(val_loss, self.model)
            self.early_stopper(val_loss)

            if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()

            if self.early_stopper.early_stop:
                if self.logger:
                    self.logger.warning(f"[EarlyStopping] Eğitim erken durduruldu.")
                break

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0

        for batch in self.train_loader:
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        if self.logger:
            self.logger.info(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}")
        return avg_loss

    def _validate(self, epoch: int) -> tuple:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                total_loss += loss.item()
                all_preds.append(outputs.cpu())
                all_targets.append(targets.cpu())

        avg_loss = total_loss / len(self.val_loader)
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        metric = self.metric_fn(preds, targets)

        if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(avg_loss)

        if self.logger:
            self.logger.info(f"[Epoch {epoch}] Val Loss: {avg_loss:.4f} | Val Metric: {metric:.4f}")

        return avg_loss, metric
