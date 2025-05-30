import torch
from model.config import ModelConfig
from model.model_builder import ModelBuilder
from model.loss_functions import HaversineLoss
from model.utils import count_parameters, get_device


def run_sample():
    print("[Model Test] Başlatılıyor...")

    config = ModelConfig()
    device = get_device()

    builder = ModelBuilder(config)
    model = builder.build().to(device)

    print(f"[Model Test] Toplam Parametre Sayısı: {count_parameters(model):,}")

    dummy_input = torch.randn(4, 3, *config.input_size).to(device)
    output = model(dummy_input)

    print(f"[Model Test] Çıktı şekli: {output.shape}")
    print(f"[Model Test] Örnek çıktı: {output}")

    dummy_target = torch.tensor([[40.0, 29.0]] * 4).to(device)
    criterion = HaversineLoss()
    loss = criterion(output, dummy_target)
    print(f"[Model Test] Haversine Loss: {loss.item():.4f}")


if __name__ == "__main__":
    run_sample()
