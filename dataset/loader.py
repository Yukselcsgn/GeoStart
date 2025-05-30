import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from dataset import config, transforms
import os
import torch  # <- KOORDİNAT TENSOR'U İÇİN EKLENDİ


class GeoDataset(Dataset):
    def __init__(self, csv_path=config.CSV_PATH, image_dir=config.IMAGE_DIR, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform if transform else transforms.default_transforms

        # Temizlik
        self.df.dropna(subset=['path', 'lat', 'lon'], inplace=True)
        self.df['full_path'] = self.df['path'].apply(lambda x: os.path.join(image_dir, x))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['full_path']).convert("RGB")
        image = self.transform(image)
        coords = torch.tensor([row['lat'], row['lon']], dtype=torch.float32)  # TEK TENSOR
        return image, coords


def get_dataloader(batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE, num_workers=config.NUM_WORKERS):
    dataset = GeoDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
