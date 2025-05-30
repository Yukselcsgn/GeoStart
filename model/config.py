import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class TrainingConfig:
    batch_size: int = 64
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    save_interval: int = 5
    eval_interval: int = 1
    early_stopping_patience: int = 10
    gradient_clip_value: float = 1.0
    seed: int = 42


@dataclass
class ModelConfig:
    model_type: str = "resnet18"  # "cnn", "transformer", vs.
    pretrained: bool = True
    input_size: Tuple[int, int] = (224, 224)
    output_dim: int = 2  # (latitude, longitude)


@dataclass
class DataConfig:
    train_csv: str = "./data/train.csv"
    val_csv: str = "./data/val.csv"
    test_csv: str = "./data/test.csv"
    image_root: str = "./images"
    augmentations: bool = True


@dataclass
class SelfLearningConfig:
    threshold: float = 0.9
    max_error_km: float = 5.0
    save_confidence: bool = True


@dataclass
class Paths:
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./outputs/checkpoints"
    logs_dir: str = "./outputs/logs"

    def create_dirs(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
