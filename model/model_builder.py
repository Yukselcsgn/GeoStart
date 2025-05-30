import torch.nn as nn
from torchvision import models
from model.config import ModelConfig


class ModelBuilder:
    def __init__(self, config: ModelConfig):
        self.config = config

    def build(self) -> nn.Module:
        if self.config.model_type == "resnet18":
            return self._build_resnet18()
        elif self.config.model_type == "resnet50":
            return self._build_resnet50()
        elif self.config.model_type == "mlp":
            return self._build_mlp()
        else:
            raise ValueError(f"Desteklenmeyen model tipi: {self.config.model_type}")

    def _build_resnet18(self) -> nn.Module:
        model = models.resnet18(pretrained=self.config.pretrained)
        model.fc = nn.Linear(model.fc.in_features, self.config.output_dim)
        return model

    def _build_resnet50(self) -> nn.Module:
        model = models.resnet50(pretrained=self.config.pretrained)
        model.fc = nn.Linear(model.fc.in_features, self.config.output_dim)
        return model

    def _build_mlp(self) -> nn.Module:
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.config.input_size[0] * self.config.input_size[1] * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, self.config.output_dim)
        )
