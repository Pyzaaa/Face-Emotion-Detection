import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm


# 1) Pobranie AffectNet z Kaggle
def download_affectnet():
    raw_dir = "data/affectnet_raw"
    yolo_dir = os.path.join(raw_dir, "YOLO_format")
    zip_path = "data/affectnet-yolo-format.zip"

    if os.path.exists(yolo_dir):
        print("[OK] Dataset już pobrany:", yolo_dir)
        return yolo_dir

    print("[INFO] Pobieranie AffectNet z Kaggle...")
    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(
        "fatihkgg/affectnet-yolo-format",
        path="data",
        unzip=False
    )

    print("[INFO] Rozpakowywanie...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(raw_dir)
    os.remove(zip_path)

    print("[OK] Dataset pobrany i rozpakowany:", yolo_dir)
    return yolo_dir


# 2) Dataset w formacie YOLO
class AffectNetYOLODataset(Dataset):
    def __init__(self, img_dir, lbl_dir, transform=None):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.transform = transform

        self.samples = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.samples.sort()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        lbl_path = os.path.join(
            self.lbl_dir,
            img_name.rsplit(".", 1)[0] + ".txt"
        )

        img = Image.open(img_path).convert("RGB")
        with open(lbl_path, "r") as f:
            label = int(f.readline().split()[0])

        if self.transform:
            img = self.transform(img)

        return img, label


# 3) Trening MobileNetV3-Large z mixed precision
def train_mobilenet_v3():
    yolo_path = download_affectnet()

    train_img_dir = os.path.join(yolo_path, "train", "images")
    train_lbl_dir = os.path.join(yolo_path, "train", "labels")
    val_img_dir = os.path.join(yolo_path, "valid", "images")
    val_lbl_dir = os.path.join(yolo_path, "valid", "labels")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    train_dataset = AffectNetYOLODataset(train_img_dir, train_lbl_dir, transform)
    val_dataset = AffectNetYOLODataset(val_img_dir, val_lbl_dir, transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Trening na urządzeniu: {device}")

    # MobileNetV3-Large z wagami z ImageNet
    model = models.mobilenet_v3_large(
        weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
    )
    # Podmiana ostatniej warstwy na 8 klas emocji
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 8)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    def evaluate(m, loader):
        m.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    outputs = m(imgs)
                _, preds = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (preds == labels).sum().item()

        return correct / total if total else 0.0

    EPOCHS = 10
    best_acc = 0.0
    os.makedirs("models", exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        loop = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{EPOCHS}",
            leave=False
        )

        for imgs, labels in loop:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        val_acc = evaluate(model, val_loader)
        avg_loss = running_loss / len(train_loader)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Avg Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/mobilenet_v3_best.pth")
            print("[✔] Zapisano nowy najlepszy model")

    print("\nTrening MobileNetV3 zakończony. Najlepsza dokładność:", best_acc)


if __name__ == "__main__":
    train_mobilenet_v3()
