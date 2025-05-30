import torch
from torchvision import models


def load_pretrained_resnet(model_name: str = "resnet18", output_dim: int = 2, freeze: bool = True):
    """
    ResNet mimarili önceden eğitilmiş model yükleyip son katmanını değiştirir.
    """
    model_fn = getattr(models, model_name, None)
    if model_fn is None:
        raise ValueError(f"Desteklenmeyen model: {model_name}")

    model = model_fn(pretrained=True)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, output_dim)

    if freeze:
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    return model


def load_custom_weights(model, path, device):
    """
    Özel bir checkpoint dosyasından model ağırlıklarını yükler.
    """
    try:
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[Pretrained] Ağırlıklar başarıyla yüklendi: {path}")
    except Exception as e:
        print(f"[Pretrained] Hata oluştu: {e}")
