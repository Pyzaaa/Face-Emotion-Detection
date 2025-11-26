import os
import torch
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from multiprocessing import freeze_support

class AffectNetYOLOTestDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, transform=None):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.transform = transform
        self.samples = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        self.samples.sort()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        lbl_path = os.path.join(self.lbl_dir, img_name.rsplit(".", 1)[0] + ".txt")
        img = Image.open(img_path).convert("RGB")
        with open(lbl_path, "r") as f:
            label = int(f.readline().split()[0])
        if self.transform:
            img = self.transform(img)
        return img, label, img_name

if __name__ == "__main__":
    freeze_support()

    test_img_dir = "data/affectnet_raw/YOLO_format/test/images"
    test_lbl_dir = "data/affectnet_raw/YOLO_format/test/labels"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = AffectNetYOLOTestDataset(test_img_dir, test_lbl_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Używane urządzenie: {device}")

    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, 8)
    model.load_state_dict(torch.load("models/vit_b16_best.pth", map_location=device))
    model.to(device)
    model.eval()

    emotion_classes = ["Anger", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    correct, total = 0, 0
    for imgs, labels, _ in tqdm(test_loader, desc="Evaluating"):
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total if total else 0.0
    print(f"\nDokładność na zbiorze testowym: {acc:.4f}")

    print("\nPrzykładowe predykcje (pierwsze 10):")
    for i in range(min(10, len(test_dataset))):
        img, label, img_name = test_dataset[i]
        with torch.no_grad():
            pred = torch.argmax(model(img.unsqueeze(0).to(device)), 1).item()
        print(f"{img_name}: prawda={emotion_classes[label]}, predykcja={emotion_classes[pred]}")
