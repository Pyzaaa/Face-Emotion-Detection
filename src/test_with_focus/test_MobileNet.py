import os
import torch
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
import cv2


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

        img_tf = self.transform(img) if self.transform else img
        return img_tf, label, img_name


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, inp, out):
        self.activations = out.detach()

    def save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self):
        # activations: (1, C, H, W), gradients: (1, C, H, W)
        weights = torch.mean(self.gradients, dim=(2, 3))[0]  # (C,)
        cam = torch.zeros(self.activations.shape[2:], device=self.activations.device)

        for i, w in enumerate(weights):
            cam += w * self.activations[0, i]

        cam = torch.clamp(cam, min=0)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-6)
        return cam.cpu().numpy()


def save_cam_overlay(img_pil, cam_map, out_path):
    img = np.array(img_pil.resize((224, 224)))

    cam_map = cv2.resize(cam_map, (224, 224))
    cam_map = np.uint8(255 * cam_map)

    heatmap = cv2.applyColorMap(cam_map, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    plt.figure(figsize=(4, 4))
    plt.imshow(overlay)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


if __name__ == "__main__":
    out_dir = "mobilenet_cam"
    os.makedirs(out_dir, exist_ok=True)

    test_img_dir = "../data/affectnet_raw/YOLO_format/test/images"
    test_lbl_dir = "../data/affectnet_raw/YOLO_format/test/labels"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = AffectNetYOLOTestDataset(test_img_dir, test_lbl_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(in_features, 8)

    model.load_state_dict(torch.load("../models/mobilenet_v3_best.pth", map_location=device))
    model.to(device)
    model.eval()

    emotion_classes = ["Anger", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    # target layer: ostatni block konwolucyjny
    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)

    all_labels, all_preds = [], []

    for imgs, labels, names in tqdm(test_loader, desc="Evaluating MobileNet + GradCAM"):
        imgs = imgs.to(device)
        labels = labels.to(device)

        for i in range(len(imgs)):
            img = imgs[i].unsqueeze(0)
            label = labels[i].item()
            name = names[i]

            img_path = os.path.join(test_img_dir, name)
            original_pil = Image.open(img_path).convert("RGB")

            model.zero_grad()
            out = model(img)
            pred = torch.argmax(out).item()

            all_labels.append(label)
            all_preds.append(pred)

            out[0, pred].backward()
            cam = gradcam.generate()

            save_path = os.path.join(out_dir, f"{name.split('.')[0]}_cam.png")
            save_cam_overlay(original_pil, cam, save_path)

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, labels=list(range(8)), average=None, zero_division=0)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)

    report = classification_report(all_labels, all_preds, target_names=emotion_classes)
    cm = confusion_matrix(all_labels, all_preds)

    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(f"accuracy={acc:.4f}\n")
        f.write(f"macro_precision={p_macro:.4f}\n")
        f.write(f"macro_recall={r_macro:.4f}\n")
        f.write(f"macro_f1={f1_macro:.4f}\n\n")
        for i, cls in enumerate(emotion_classes):
            f.write(f"{cls}: P={precision[i]:.3f}, R={recall[i]:.3f}, F1={f1[i]:.3f}\n")

    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write(report)

    np.savetxt(os.path.join(out_dir, "confusion_matrix.csv"), cm, delimiter=",", fmt="%d")

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap="viridis")
    plt.title("Confusion matrix - MobileNet GradCAM")
    plt.colorbar()
    ticks = np.arange(len(emotion_classes))
    plt.xticks(ticks, emotion_classes, rotation=45, ha="right")
    plt.yticks(ticks, emotion_classes)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()

    print("\n[OK] Wyniki + GradCAM zapisane")
