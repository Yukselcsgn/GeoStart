# Model Paketi

Bu paket, coğrafi konum tahmini ve kendinden öğrenme (self-learning) amaçlı derin öğrenme modelleri için kapsamlı, modüler ve gelişmiş bir yapıdır. Türkiye gibi dinamik şehir yapılarında değişen sokak ve koordinat verileri üzerinde çalışmak üzere optimize edilmiştir.

---

## İçerik ve Modüller

| Dosya Adı             | Açıklama                                                                                         |
|-----------------------|-------------------------------------------------------------------------------------------------|
| `config.py`           | Modelin hiperparametreleri ve yapılandırma ayarlarının merkezi dosyası.                         |
| `model_builder.py`    | CNN, Transformer gibi farklı model mimarilerinin oluşturulduğu modül.                          |
| `pretrained_models.py`| Önceden eğitilmiş modellerin yüklenmesi ve transfer öğrenme için adaptasyon fonksiyonları.     |
| `utils.py`            | Model ağırlıklarının kaydedilmesi/yüklenmesi, cihaz yönetimi, parametre sayımı, resim yükleme.  |
| `model_main.py`       | Paket bağımsız testler, örnek model oluşturma ve ileriye (forward) geçişin çalıştırılması.       |
| `loss_functions.py`   | Proje için özelleştirilmiş kayıp fonksiyonları.                                                 |
| `metrics.py`          | Performans ölçümü için jeodezik mesafe gibi özel metriklerin tanımlandığı dosya.                |
| `self_learning.py`    | Kendinden öğrenme (pseudo-labeling) algoritması ve sınıfı.                                      |

---

## Özellikler

- **Modüler ve Genişletilebilir:** Projede her fonksiyon ve sınıf kendi dosyasında yer alır, kolayca özelleştirilebilir.
- **Kendinden Öğrenme (Pseudo-labeling):** Etiketlenmemiş verilerde yüksek güvenlikli tahminler üreterek modelin kendini geliştirmesini sağlar.
- **Jeodezik Mesafe Hesaplama:** Koordinatlar arasındaki gerçek dünya mesafesini hassas şekilde ölçer.
- **Esnek Model Desteği:** CNN ve Transformer tabanlı modeller kolayca entegre edilebilir.
- **Cihaz Uyumluluğu:** CPU veya GPU otomatik algılanır ve kullanılır.
- **Checkpoint Mekanizması:** Model ağırlıkları güvenli şekilde kaydedilir ve yüklenir.
- **Hata Yönetimi:** Yükleme, kayıt ve veri işleme sırasında kapsamlı hata yakalama ve bilgilendirme.
- **Performans Takibi:** Özel metrikler ile eğitim ve doğrulama süreçleri takip edilir.

---

## Kurulum

1. Gerekli kütüphaneleri yükleyin:

```bash
pip install torch torchvision pillow numpy geopy tqdm
```

2. Model paketini projenize ekleyin.

---

## Kullanım

### Model İnşası ve Test

```python
from model.model_builder import build_model
from model.utils import get_device, load_checkpoint, save_checkpoint

device = get_device()
model = build_model().to(device)

# Model ağırlıklarını yükle
load_checkpoint(model, "model_checkpoint.pth", device)

# Model ağırlıklarını kaydet
save_checkpoint(model, "model_checkpoint.pth")
```

### Kendinden Öğrenme ile Pseudo-label Üretimi

```python
from model.self_learning import SelfLearner
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

self_learner = SelfLearner(model, transform, threshold=0.9, max_error_km=5.0, device=device)
pseudo_labels = self_learner.generate_pseudo_labels(unlabeled_dataframe, save_confidence=True)
```

---

## Kod Yapısı ve Geliştirme Rehberi

- Yeni modeller `model_builder.py` içine eklenebilir.
- Özel kayıp fonksiyonları `loss_functions.py` dosyasına dahil edilmelidir.
- Performans metrikleri `metrics.py` üzerinden takip edilmelidir.
- Kendinden öğrenme algoritması `self_learning.py` içerisinde modüler bir şekilde geliştirilmiştir.
- Yardımcı fonksiyonlar için `utils.py` kullanılır.

---

## Katkıda Bulunma

- Katkılarınız için teşekkür ederiz!  
- Yeni özellikler, hata düzeltmeleri ve iyileştirmeler için lütfen `issue` açın veya `pull request` gönderin.  
- Kod yazarken PEP8 standartlarına uymaya, yeterli yorum ve dokümantasyon sağlamaya özen gösteriniz.  

---

## Lisans

Bu proje MIT Lisansı kapsamında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakınız.

---

## İletişim

Sorularınız, önerileriniz veya destek talepleriniz için iletişime geçebilirsiniz.

--- 

© 2025 | Hazırlayan: **[Yüksel COŞGUN]** 
