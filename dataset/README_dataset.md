# 📦 GeoDataset Package

Bu modül, coğrafi konum bilgileri (enlem/boylam) ile ilişkilendirilmiş görsellerin PyTorch ortamında kolayca yüklenmesini ve işlenmesini sağlayan bir dataset paketidir.

## 📁 Proje Yapısı

```
dataset/
│
├── config.py          # Dizinler ve dataloader ayarları
├── loader.py          # GeoDataset sınıfı ve dataloader fonksiyonu
├── transforms.py      # Görüntü işleme pipeline'ı
└── dataset_main.py    # Veri kümesinin test edilmesini sağlayan örnek main dosyası
```

## 🧠 Kullanım Amacı

Makine öğrenimi modellerinde, özellikle **coğrafi regresyon**, **lokasyon tahmini**, **uydu görüntü analizi** gibi alanlarda konum bilgisi içeren görüntülerin yüklenmesi, işlenmesi ve batch'lenmesi amacıyla kullanılır.

## 🚀 Özellikler

- CSV dosyasından yol, enlem ve boylam bilgilerini okur.
- Görüntüleri otomatik olarak normalize eder ve yeniden boyutlandırır.
- `torch.utils.data.DataLoader` ile uyumlu `GeoDataset` sınıfı içerir.
- Eksik ya da bozuk verileri otomatik temizler.
- Her örnek: `(image_tensor, [lat, lon])` formatında döner.
- Batch’li çıktı: `(B, 3, 224, 224)` ve `(B, 2)` boyutunda olur.

## 🔧 Kurulum

1. Gerekli kütüphaneleri yükleyin:

```bash
pip install torch torchvision pandas pillow
```

2. `dataset/` klasörünü projenize dahil edin veya bağımsız kullanın.

## 📂 Girdi Formatı

### CSV (metadata.csv)

```csv
path,lat,lon
img_001.jpg,39.9901,32.6913
img_002.jpg,39.9910,32.6919
...
```

- `path`: Görselin `images/` klasörü içindeki göreli yolu
- `lat`: Enlem (float)
- `lon`: Boylam (float)

### Görseller

Tüm görseller `images/` klasörü içinde yer almalıdır.

## 🔍 Örnek Kullanım (`dataset_main.py`)

```python
from dataset.loader import get_dataloader

loader = get_dataloader()
for images, coords in loader:
    print(images.shape)  # torch.Size([B, 3, 224, 224])
    print(coords.shape)  # torch.Size([B, 2])
    break
```

## ⚙️ Özelleştirme

### Dataloader Ayarları (`config.py`)
```python
BATCH_SIZE = 32
NUM_WORKERS = 4
SHUFFLE = True
```

### Görüntü İşleme (`transforms.py`)
```python
default_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

İsteğe bağlı olarak `transform` parametresi `GeoDataset`'e geçilebilir.

## 🧪 Test

```bash
python dataset/dataset_main.py
```

Beklenen çıktı:
```
Testing GeoDataset loading...
Batch 1:
 - Image batch shape: torch.Size([32, 3, 224, 224])
 - Coordinates batch shape: torch.Size([32, 2])
 - First 2 coordinates:
tensor([[39.9901, 32.6913],
        [39.9910, 32.6919]])
```

## 🤝 Katkıda Bulunma

Katkılar memnuniyetle karşılanır! Lütfen yeni özellikler eklemeden önce bir "issue" oluşturun. Kod katkıları için:

1. Fork'la
2. Yeni bir branch oluştur (`feature/my-new-feature`)
3. Değişiklik yap ve commit et
4. Pull request gönder

## 📝 Lisans

MIT Lisansı

---
 
© 2025 | Hazırlayan: **[Yüksel COŞGUN]** 