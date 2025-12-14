import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score

torch.backends.cudnn.benchmark = True


# ======================================================
# 1. DOWNLOAD DATA
# ======================================================
def download_affectnet():
    raw_dir = "../data/affectnet_raw"
    yolo_dir = os.path.join(raw_dir, "YOLO_format")
    zip_path = "../data/affectnet-yolo-format.zip"

    if os.path.exists(yolo_dir):
        print("[OK] Dataset already downloaded:", yolo_dir)
        return yolo_dir

    print("[INFO] Downloading AffectNet from Kaggle...")
    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(
        "fatihkgg/affectnet-yolo-format",
        path="data",
        unzip=False
    )

    print("[INFO] Unzipping...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(raw_dir)

    os.remove(zip_path)
    print("[OK] Dataset extracted:", yolo_dir)
    return yolo_dir


# ======================================================
# 2. DATASET
# ======================================================
class AffectNetYOLODataset(Dataset):
    def __init__(self, img_dir, lbl_dir, transform=None):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.transform = transform

        all_imgs = [f for f in os.listdir(img_dir)
                    if f.endswith(".png") or f.endswith(".jpg")]

        self.samples = []
        for f in tqdm(all_imgs, desc=f"Loading {os.path.basename(img_dir)}"):
            self.samples.append((f, os.path.join(lbl_dir, f.rsplit(".", 1)[0] + ".txt")))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, lbl_path = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        with open(lbl_path, "r") as f:
            cls = int(f.readline().split()[0])

        if cls < 0 or cls > 7:
            cls = 0  # safety fallback

        if self.transform:
            img = self.transform(img)

        return img, cls


# ======================================================
# 3. TRAIN / EVAL LOOPS
# ======================================================
def evaluate(model, loader, device):
    model.eval()
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(imgs)

            _, preds = torch.max(outputs, 1)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    acc = (torch.tensor(preds_all) == torch.tensor(labels_all)).float().mean().item()
    f1 = f1_score(labels_all, preds_all, average="macro")
    return acc, f1


# ======================================================
# 4. MAIN TRAINING FUNCTION
# ======================================================
def train_resnet50_improved():
    yolo_path = download_affectnet()

    # Folders
    train_img = os.path.join(yolo_path, "train", "images")
    train_lbl = os.path.join(yolo_path, "train", "labels")
    val_img = os.path.join(yolo_path, "valid", "images")
    val_lbl = os.path.join(yolo_path, "valid", "labels")

    # Stronger augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Datasets
    train_dataset = AffectNetYOLODataset(train_img, train_lbl, train_transform)
    val_dataset = AffectNetYOLODataset(val_img, val_lbl, val_transform)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,
                            num_workers=4, pin_memory=True)

    device = torch.device("cuda")
    print("[INFO] Training on:", device)

    # ======================================================
    # Build model
    # ======================================================
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 8)
    model = model.to(device)

    # ======================================================
    # PHASE 1 — train only classifier head
    # ======================================================
    print("\n[PHASE 1] Training classifier head only...")

    for param in model.parameters():
        param.requires_grad = False
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    scaler = torch.cuda.amp.GradScaler()

    best_f1 = 0
    patience = 5
    patience_counter = 0

    # Phase 1 training
    for epoch in range(5):
        model.train()
        running_loss = 0

        loop = tqdm(train_loader, desc=f"[Phase1] Epoch {epoch+1}/5", leave=False)
        for imgs, labels in loop:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        scheduler.step()

        # Validation
        acc, f1 = evaluate(model, val_loader, device)
        print(f"[Phase1] Epoch {epoch+1} | Loss={running_loss/len(train_loader):.4f} | "
              f"Acc={acc:.4f} | F1={f1:.4f}")

    # ======================================================
    # PHASE 2 — unfreeze entire model
    # ======================================================
    print("\n[PHASE 2] Fine-tuning whole network...")

    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

    best_f1 = 0
    patience_counter = 0

    # Train 25 epochs
    for epoch in range(25):
        model.train()
        running_loss = 0

        loop = tqdm(train_loader, desc=f"[Phase2] Epoch {epoch+1}/25", leave=False)
        for imgs, labels in loop:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        scheduler.step()

        acc, f1 = evaluate(model, val_loader, device)
        print(f"[Phase2] Epoch {epoch+1} | Loss={running_loss/len(train_loader):.4f} | "
              f"Acc={acc:.4f} | F1={f1:.4f}")

        # Early stopping
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "../models/resnet50_improved.pth")
            print("[OK] Saved new best model (F1 improved)")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("[STOP] Early stopping triggered.")
            break

    print("\nTraining finished!")
    print("Best validation F1:", best_f1)


# ======================================================
# RUN
# ======================================================
if __name__ == "__main__":
    train_resnet50_improved()
