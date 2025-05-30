import os
import torch
from torchvision import transforms
from model.config import ModelConfig, TrainingConfig, Paths
from model.model_builder import ModelBuilder
from model.metrics import compute_haversine_distance
from dataset.loader import get_dataloader
from dataset.transforms import get_transform
from training import (
    Trainer,
    get_device,
    set_seed,
    get_logger,
    get_optimizer,
    get_scheduler,
    get_loss_function
)


def main():
    # === CONFIGURATION ===
    model_config = ModelConfig()
    train_config = TrainingConfig()
    paths = Paths()
    paths.create_dirs()

    # === SETUP ===
    set_seed(train_config.seed)
    device = get_device()
    logger = get_logger(paths.logs_dir)
    logger.info("[MAIN] Eğitim konfigürasyonu yüklendi.")

    # === DATA LOADERS ===
    augment = train_config.num_epochs > 20
    transform = get_transform(train=True, augment=augment)
    train_loader, val_loader = get_dataloader(transform=transform, batch_size=train_config.batch_size)
    logger.info(f"[MAIN] Eğitim ve doğrulama verisi yüklendi. Batch size: {train_config.batch_size}")

    # === MODEL ===
    model = ModelBuilder(model_config).build()
    loss_fn = get_loss_function(train_config.loss_function, alpha=train_config.loss_alpha)
    metric_fn = compute_haversine_distance
    optimizer = get_optimizer(
        train_config.optimizer_name,
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay
    )
    scheduler = get_scheduler(
        optimizer,
        scheduler_name=train_config.scheduler_name,
        T_max=train_config.num_epochs
    )

    # === TRAINING ===
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        metric_fn=metric_fn,
        device=device,
        scheduler=scheduler,
        logger=logger,
        checkpoint_path=os.path.join(paths.checkpoint_dir, "best_model.pth"),
        early_stopping_patience=train_config.early_stopping_patience
    )

    logger.info("[MAIN] Eğitim başlatılıyor...")
    trainer.train(num_epochs=train_config.num_epochs)
    logger.info("[MAIN] Eğitim tamamlandı.")


if __name__ == "__main__":
    main()
