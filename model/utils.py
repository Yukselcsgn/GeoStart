# utils.py

import torch
from PIL import Image
import os

def load_image(image_path, transform, device):
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Dosya bulunamadı: {image_path}")
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).to(device)
        return tensor
    except Exception as e:
        print(f"[load_image] Hata: {e} - Dosya: {image_path}")
        return None

def save_model(model, path):
    try:
        torch.save(model.state_dict(), path)
        print(f"[Checkpoint] Model kaydedildi: {path}")
    except Exception as e:
        print(f"[Checkpoint] Kaydetme hatası: {e}")

def load_model(model, path, device):
    try:
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[Checkpoint] Model başarıyla yüklendi: {path}")
    except Exception as e:
        print(f"[Checkpoint] Yükleme hatası: {e}")

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
