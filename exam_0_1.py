import os
import requests
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import pandas as pd
from io import BytesIO

# === CONFIG ===
API_KEY = "AIzaSyDHRY_3umhvE9gE4zhecyxH9oeDQ9rl1_4"
IMAGE_DIR = "images/"
METADATA_FILE = "metadata.csv"
IMG_SIZE = 224
BASE_LAT = 39.990189
BASE_LON = 32.690970

COORDS = [
    (BASE_LAT, BASE_LON),
    (BASE_LAT + 0.0001, BASE_LON),
    (BASE_LAT - 0.0001, BASE_LON),
    (BASE_LAT, BASE_LON + 0.0001),
    (BASE_LAT, BASE_LON - 0.0001),
    (BASE_LAT + 0.0001, BASE_LON + 0.0001),
    (BASE_LAT - 0.0001, BASE_LON - 0.0001),
    (BASE_LAT + 0.0002, BASE_LON),
    (BASE_LAT - 0.0002, BASE_LON),
    (BASE_LAT, BASE_LON + 0.0002),
    (BASE_LAT, BASE_LON - 0.0002),
]

HEADINGS = [0, 90, 180, 270]

# === STEP 1: Image Download ===
def download_streetview_images():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    metadata = []
    for i, (lat, lon) in enumerate(COORDS):
        for h in HEADINGS:
            filename = f"img_{i}_h{h}.jpg"
            path = os.path.join(IMAGE_DIR, filename)

            if os.path.exists(path):
                print(f"[INFO] Zaten mevcut: {path}")
                metadata.append((path, lat, lon, h))
                continue

            url = f"https://maps.googleapis.com/maps/api/streetview?size=640x640&location={lat},{lon}&heading={h}&fov=90&pitch=0&key={API_KEY}"
            response = requests.get(url)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                img.save(path)
                metadata.append((path, lat, lon, h))
                print(f"[✓] İndirildi: {path}")
            else:
                print(f"[✗] İndirme hatası: {lat}, {lon}, heading: {h}, Status: {response.status_code}")

    if metadata:
        pd.DataFrame(metadata, columns=["path", "lat", "lon", "heading"]).to_csv(METADATA_FILE, index=False)
        print(f"[INFO] metadata.csv oluşturuldu. Kayıt sayısı: {len(metadata)}")
    else:
        raise ValueError("Görsel verisi oluşturulamadı.")

# === STEP 2: Dataset ===
class GeoDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        img = self.transform(img)
        coords = torch.tensor([row['lat'], row['lon']], dtype=torch.float32)
        return img, coords

# === STEP 3: Model ===
class GeoCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet18(pretrained=True)
        self.base.fc = nn.Linear(self.base.fc.in_features, 2)  # latitude, longitude

    def forward(self, x):
        return self.base(x)

# === STEP 4: Train ===
def train(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for imgs, coords in dataloader:
        imgs, coords = imgs.to(device), coords.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = loss_fn(preds, coords)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# === STEP 5: Eval ===
def evaluate(model, dataloader, device):
    model.eval()
    errors = []
    with torch.no_grad():
        for imgs, coords in dataloader:
            imgs = imgs.to(device)
            preds = model(imgs).cpu().numpy()
            targets = coords.numpy()
            for p, t in zip(preds, targets):
                err = geodesic((p[0], p[1]), (t[0], t[1])).km
                errors.append(err)
    return sum(errors) / len(errors)

def predict_image(image_path, model, device):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(img_tensor).cpu().numpy()[0]
    return pred  # [latitude, longitude]


def predict_from_file():
    model_path = "model.pth"
    image_path = input("Tahmin etmek istediğiniz görüntünün yolunu girin: ").strip()

    if not os.path.exists(image_path):
        print("[HATA] Dosya bulunamadı:", image_path)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeoCNN().to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("[✓] Model yüklendi.")
    else:
        print("[HATA] Eğitilmiş model bulunamadı. Lütfen önce eğitin (main()).")
        return

    pred = predict_image(image_path, model, device)
    print(f"[📍] Tahmin edilen konum: Enlem: {pred[0]:.6f}, Boylam: {pred[1]:.6f}")

# === MAIN ===
def main():
    if not os.path.exists(METADATA_FILE):
        print("[INFO] Görseller indiriliyor...")
        download_streetview_images()

    df = pd.read_csv(METADATA_FILE)
    if df.empty:
        print("[ERROR] metadata.csv boş. Görseller indirilememiş olabilir.")
        download_streetview_images()
        df = pd.read_csv(METADATA_FILE)
        if df.empty:
            raise ValueError("metadata.csv hâlâ boş. Devam edilemiyor.")

    print(f"[INFO] Toplam veri sayısı: {len(df)}")
    train_df, test_df = train_test_split(df, test_size=0.2)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    train_ds = GeoDataset(train_df, transform)
    test_ds = GeoDataset(test_df, transform)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeoCNN().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(5):
        loss = train(model, train_dl, loss_fn, optimizer, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    error_km = evaluate(model, test_dl, device)
    print(f"Ortalama konum hatası: {error_km:.2f} km")

    # Eğitim sonrası model kaydı
    torch.save(model.state_dict(), "model.pth")
    print("[✓] Model kaydedildi: model.pth")

    # Dışarıdan fotoğrafla tahmin
    predict_from_file()


if __name__ == "__main__":
    main()
