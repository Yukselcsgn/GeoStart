import torch
from dataset.loader import get_dataloader

def test_dataset():
    loader = get_dataloader()
    for i, (images, coords) in enumerate(loader):
        print(f"Batch {i+1}:")
        print(f" - Image batch shape: {images.shape}")  # [B, 3, 224, 224]
        print(f" - Coordinates batch shape: {coords.shape}")  # [B, 2]
        print(f" - First 2 coordinates:\n{coords[:2]}")
        if i == 0:
            break

if __name__ == "__main__":
    print("Testing GeoDataset loading...")
    test_dataset()
