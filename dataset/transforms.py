# geo_dataset/transforms.py

from torchvision import transforms

# Görüntüleri normalize etme ve yeniden boyutlandırma
default_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet standartları
                         std=[0.229, 0.224, 0.225])
])
