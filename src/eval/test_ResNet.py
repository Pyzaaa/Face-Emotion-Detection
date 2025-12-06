import os
import torch
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from multiprocessing import freeze_support

# -------------------------------
# 1. Dataset testowy
# -------------------------------
class AffectNetYOLOTestDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, transform=None):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.transform = transform
        self.samples = [f for f in os.listdir(img_dir) if f.endswith(".png") or f.endswith(".jpg")]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        lbl_path = os.path.join(self.lbl_dir, img_name.rsplit(".",1)[0] + ".txt")

        img = Image.open(img_path).convert("RGB")
        with open(lbl_path, "r") as f:
            label = int(f.readline().split()[0])

        if self.transform:
            img = self.transform(img)

        return img, label, img_name


# -------------------------------
# 2. Start programu
# -------------------------------
if __name__ == "__main__":
    freeze_support()  # wymagane na Windows, jeśli używamy DataLoader

    # Ścieżki do testowego zbioru
    test_img_dir = "data/affectnet_raw/YOLO_format/test/images"
    test_lbl_dir = "data/affectnet_raw/YOLO_format/test/labels"

    # Transformacje takie same jak podczas treningu
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # Dataset i DataLoader
    test_dataset = AffectNetYOLOTestDataset(test_img_dir, test_lbl_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)  # 0 dla Windows

    # Wczytanie modelu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Używane urządzenie: {device}")

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Linear(model.fc.in_features, 8)  # 8 emocji
    model.load_state_dict(torch.load("models/resnet50_best.pth", map_location=device))
    model.to(device)
    model.eval()

    # Klasy emocji
    emotion_classes = ["Anger", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    # -------------------------------
    # 3. Ewaluacja
    # -------------------------------
    correct = 0
    total = 0

    for imgs, labels, _ in tqdm(test_loader, desc="Evaluating"):
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    print(f"\nDokładność na zbiorze testowym: {accuracy:.4f}")

    # -------------------------------
    # 4. Przykładowe predykcje
    # -------------------------------
    print("\nPrzykładowe predykcje (pierwsze 10 obrazów):")
    for i in range(min(10, len(test_dataset))):
        img, label, img_name = test_dataset[i]
        img_tensor = img.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.argmax(output, dim=1).item()
        print(f"{img_name}: prawda={emotion_classes[label]}, predykcja={emotion_classes[pred]}")
