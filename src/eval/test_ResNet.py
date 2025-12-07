import os
import torch
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from multiprocessing import freeze_support

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import numpy as np

class AffectNetYOLOTestDataset(Dataset):
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
        lbl_path = os.path.join(self.lbl_dir, img_name.rsplit(".", 1)[0] + ".txt")

        img = Image.open(img_path).convert("RGB")
        with open(lbl_path, "r") as f:
            label = int(f.readline().split()[0])

        if self.transform:
            img = self.transform(img)

        return img, label, img_name

if __name__ == "__main__":
    freeze_support() 

    results_dir = "experiments/baseline/resnet"
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "metrics.txt")
    report_path = os.path.join(results_dir, "classification_report.txt")
    cm_png_path = os.path.join(results_dir, "confusion_matrix.png")
    cm_csv_path = os.path.join(results_dir, "confusion_matrix.csv")

    test_img_dir = "data/affectnet_raw/YOLO_format/test/images"
    test_lbl_dir = "data/affectnet_raw/YOLO_format/test/labels"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_dataset = AffectNetYOLOTestDataset(test_img_dir, test_lbl_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Używane urządzenie: {device}")

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Linear(model.fc.in_features, 8)  # 8 emocji
    model.load_state_dict(torch.load("models/resnet50_best.pth", map_location=device))
    model.to(device)
    model.eval()

    emotion_classes = [
        "Anger", "Contempt", "Disgust", "Fear",
        "Happy", "Neutral", "Sad", "Surprise"
        ]

    all_labels = []
    all_preds = []

    for imgs, labels, _ in tqdm(test_loader, desc="Evaluating ResNet50"):
        imgs = imgs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(8)), average=None, zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    print(f"\nDokładność ResNet50 na zbiorze testowym: {acc:.4f}")
    print(f"Macro precision: {p_macro:.4f}")
    print(f"Macro recall:    {r_macro:.4f}")
    print(f"Macro F1:        {f1_macro:.4f}")

    with open(metrics_path, "w") as f:
        f.write("ResNet50 baseline - test metrics\n")
        f.write(f"accuracy={acc:.4f}\n")
        f.write(f"macro_precision={p_macro:.4f}\n")
        f.write(f"macro_recall={r_macro:.4f}\n")
        f.write(f"macro_f1={f1_macro:.4f}\n\n")
        f.write("per-class metrics (order: Anger, Contempt, Disgust, Fear, Happy, Neutral, Sad, Surprise)\n")
        for i, cls_name in enumerate(emotion_classes):
            f.write(
                f"{cls_name}: precision={precision[i]:.4f}, "
                f"recall={recall[i]:.4f}, f1={f1[i]:.4f}\n"
            )

    report = classification_report(
        all_labels, all_preds,
        target_names=emotion_classes,
        zero_division=0
    )
    with open(report_path, "w") as f:
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(8)))
    np.savetxt(cm_csv_path, cm, delimiter=",", fmt="%d")

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap="viridis")
    plt.title("Confusion matrix - ResNet50 baseline")
    plt.colorbar()

    tick_marks = np.arange(len(emotion_classes))
    plt.xticks(tick_marks, emotion_classes, rotation=45, ha="right")
    plt.yticks(tick_marks, emotion_classes)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9
            )

    plt.xlabel("Predykcja")
    plt.ylabel("Prawda")
    plt.tight_layout()
    plt.savefig(cm_png_path)
    plt.close()

    print(f"\n[OK] Zapisano metryki do: {metrics_path}")
    print(f"[OK] Zapisano classification report do: {report_path}")
    print(f"[OK] Zapisano confusion matrix do: {cm_png_path}")

    print("\nPrzykładowe predykcje (pierwsze 10 obrazów):")
    for i in range(min(10, len(test_dataset))):
        img, label, img_name = test_dataset[i]
        img_tensor = img.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.argmax(output, dim=1).item()
        print(
            f"{img_name}: prawda={emotion_classes[label]}, "
            f"predykcja={emotion_classes[pred]}"
        )
