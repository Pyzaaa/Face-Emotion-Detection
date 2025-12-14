import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# ----------------------------------------------------
# Dataset
# ----------------------------------------------------
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
            img_tf = self.transform(img)
        else:
            img_tf = img

        return img_tf, label, img_name, img


# ----------------------------------------------------
# GRAD-CAM IMPLEMENTACJA
# ----------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # rejestracja hooków
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, class_idx):
        # Grad-CAM
        weights = torch.mean(self.gradients, dim=(1, 2))  # GAP po HxW

        grad_cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32).to(self.activations.device)
        for i, w in enumerate(weights):
            grad_cam += w * self.activations[0, i]

        grad_cam = torch.clamp(grad_cam, min=0)
        grad_cam = grad_cam - grad_cam.min()
        grad_cam = grad_cam / (grad_cam.max() + 1e-6)

        return grad_cam.cpu().numpy()


# ----------------------------------------------------
# Heatmap overlay
# ----------------------------------------------------
def save_gradcam_overlay(img_pil, gradcam_map, out_path):
    img = np.array(img_pil.resize((224, 224)))
    heatmap = plt.cm.jet(gradcam_map)[..., :3] * 255
    heatmap = heatmap.astype(np.uint8)

    overlay = (0.4 * heatmap + 0.6 * img).astype(np.uint8)

    plt.figure(figsize=(4, 4))
    plt.imshow(overlay)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
if __name__ == "__main__":
    os.makedirs("experiments/resnet_cam", exist_ok=True)

    test_img_dir = "data/affectnet_raw/YOLO_format/test/images"
    test_lbl_dir = "data/affectnet_raw/YOLO_format/test/labels"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_dataset = AffectNetYOLOTestDataset(test_img_dir, test_lbl_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------------------
    # ŁADOWANIE MODELU
    # ----------------------------------------------------
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 8)
    model.load_state_dict(torch.load("models/resnet50_best.pth", map_location=device))
    model.to(device)
    model.eval()

    emotion_classes = [
        "Anger", "Contempt", "Disgust", "Fear",
        "Happy", "Neutral", "Sad", "Surprise"
    ]

    # Grad-CAM dla ostatniej warstwy konwolucyjnej
    target_layer = model.layer4[-1]
    gradcam = GradCAM(model, target_layer)

    all_labels = []
    all_preds = []

    # ----------------------------------------------------
    # TESTOWANIE + GENEROWANIE GRADCAM
    # ----------------------------------------------------
    for imgs, labels, names, originals in tqdm(test_loader, desc="Evaluating"):
        imgs = imgs.to(device)
        labels = labels.to(device)

        for i in range(len(imgs)):
            img = imgs[i].unsqueeze(0)
            label = labels[i].item()
            name = names[i]
            original_pil = originals[i]

            model.zero_grad()
            out = model(img)
            pred = torch.argmax(out).item()

            # zapis predykcji
            all_labels.append(label)
            all_preds.append(pred)

            # obliczenie grad-cam dla klasy predykowanej
            out[0, pred].backward(retain_graph=True)
            cam = gradcam.generate(class_idx=pred)

            save_path = f"experiments/resnet_cam/{name.replace('.jpg', '').replace('.png','')}_cam.png"
            save_gradcam_overlay(original_pil, cam, save_path)

    # ----------------------------------------------------
    # METRYKI
    # ----------------------------------------------------
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(8)), average=None, zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    print(f"\nAccuracy:      {acc:.4f}")
    print(f"Macro Precision: {p_macro:.4f}")
    print(f"Macro Recall:    {r_macro:.4f}")
    print(f"Macro F1:        {f1_macro:.4f}")

    # confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # raport
    report = classification_report(all_labels, all_preds, target_names=emotion_classes)

    with open("experiments/resnet_cam/metrics.txt", "w") as f:
        f.write(f"accuracy={acc:.4f}\n")
        f.write(f"macro_precision={p_macro:.4f}\n")
        f.write(f"macro_recall={r_macro:.4f}\n")
        f.write(f"macro_f1={f1_macro:.4f}\n")
        f.write("\nPer-class metrics:\n")
        for i, cls in enumerate(emotion_classes):
            f.write(f"{cls}: P={precision[i]:.3f}, R={recall[i]:.3f}, F1={f1[i]:.3f}\n")

    with open("experiments/resnet_cam/report.txt", "w") as f:
        f.write(report)

    print("\n[OK] Wyniki + GradCAM zapisane w: experiments/resnet_cam/")
