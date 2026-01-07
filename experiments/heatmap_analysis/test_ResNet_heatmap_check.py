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

        img_tf = self.transform(img) if self.transform else img
        return img_tf, label, img_name


# ----------------------------------------------------
# GRAD-CAM
# ----------------------------------------------------
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

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self):
        grads = self.gradients[0]   # (C,H,W)
        acts = self.activations[0]  # (C,H,W)

        weights = grads.mean(dim=(1, 2))  # (C,)
        cam = (weights[:, None, None] * acts).sum(dim=0)  # (H,W)
        cam = torch.relu(cam)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-6)
        return cam.cpu().numpy()


# ----------------------------------------------------
# Overlay + masking
# ----------------------------------------------------
def save_gradcam_overlay(img_pil, gradcam_map, out_path):
    img = np.array(img_pil.resize((224, 224)))

    gradcam_map = cv2.resize(gradcam_map, (224, 224))
    gradcam_map = np.uint8(255 * gradcam_map)

    heatmap = cv2.applyColorMap(gradcam_map, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    plt.figure(figsize=(4, 4))
    plt.imshow(overlay)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def center_mask_tensor(img_tensor, keep_ratio=0.6):
    """
    img_tensor: (1,3,224,224) po normalizacji
    Zostawia środek keep_ratio, resztę ustawia na 0 (czyli "czarne" w przestrzeni znormalizowanej).
    """
    x = img_tensor.clone()
    _, _, H, W = x.shape
    kh = int(H * keep_ratio)
    kw = int(W * keep_ratio)
    y0 = (H - kh) // 2
    x0 = (W - kw) // 2

    mask = torch.zeros((H, W), device=x.device, dtype=x.dtype)
    mask[y0:y0+kh, x0:x0+kw] = 1.0
    mask = mask[None, None, :, :]  # (1,1,H,W)

    x = x * mask
    return x


def gradcam_for_class(model, gradcam, img_tensor, class_idx):
    model.zero_grad(set_to_none=True)
    out = model(img_tensor)
    out[0, class_idx].backward()
    cam = gradcam.generate()
    return out, cam


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
if __name__ == "__main__":
    # gdzie zapisujemy
    base_dir = "experiments/heatmap_analysis/resnet_check"
    out_pred = os.path.join(base_dir, "pred")
    out_top2 = os.path.join(base_dir, "top2")
    out_mask = os.path.join(base_dir, "masked_pred")
    os.makedirs(out_pred, exist_ok=True)
    os.makedirs(out_top2, exist_ok=True)
    os.makedirs(out_mask, exist_ok=True)

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

    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 8)
    model.load_state_dict(torch.load("models/resnet50_best.pth", map_location=device))
    model.to(device)
    model.eval()

    emotion_classes = [
        "Anger", "Contempt", "Disgust", "Fear",
        "Happy", "Neutral", "Sad", "Surprise"
    ]

    target_layer = model.layer4[-1]
    gradcam = GradCAM(model, target_layer)

    all_labels, all_preds = [], []

    for imgs, labels, names in tqdm(test_loader, desc="Evaluating + GradCAM checks"):
        imgs = imgs.to(device)
        labels = labels.to(device)

        for i in range(len(imgs)):
            img = imgs[i].unsqueeze(0)  # (1,3,224,224)
            label = labels[i].item()
            name = names[i]
            stem = name.rsplit(".", 1)[0]

            img_path = os.path.join(test_img_dir, name)
            original_pil = Image.open(img_path).convert("RGB")

            # forward (bez gradów) żeby wybrać pred i top2
            with torch.no_grad():
                logits = model(img)
                probs = torch.softmax(logits, dim=1)[0]
                top2 = torch.topk(probs, k=2).indices.tolist()
                pred = top2[0]
                second = top2[1]

            all_labels.append(label)
            all_preds.append(pred)

            # CAM dla pred
            _, cam_pred = gradcam_for_class(model, gradcam, img, pred)
            save_gradcam_overlay(original_pil, cam_pred, os.path.join(out_pred, f"{stem}_pred_{pred}.png"))

            # CAM dla top2 (wrong-class check)
            _, cam_top2 = gradcam_for_class(model, gradcam, img, second)
            save_gradcam_overlay(original_pil, cam_top2, os.path.join(out_top2, f"{stem}_top2_{second}.png"))

            # CAM na obrazie po maskingu (tło wycięte)
            img_masked = center_mask_tensor(img, keep_ratio=0.6)
            _, cam_mask = gradcam_for_class(model, gradcam, img_masked, pred)
            save_gradcam_overlay(original_pil, cam_mask, os.path.join(out_mask, f"{stem}_masked_pred_{pred}.png"))

    # metryki klasyczne
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(8)), average=None, zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=emotion_classes, zero_division=0)

    metrics_path = os.path.join(base_dir, "metrics.txt")
    report_path = os.path.join(base_dir, "report.txt")
    cm_path = os.path.join(base_dir, "confusion_matrix.csv")

    with open(metrics_path, "w") as f:
        f.write(f"accuracy={acc:.4f}\n")
        f.write(f"macro_precision={p_macro:.4f}\n")
        f.write(f"macro_recall={r_macro:.4f}\n")
        f.write(f"macro_f1={f1_macro:.4f}\n\n")
        for i, cls in enumerate(emotion_classes):
            f.write(f"{cls}: P={precision[i]:.3f}, R={recall[i]:.3f}, F1={f1[i]:.3f}\n")

    with open(report_path, "w") as f:
        f.write(report)

    np.savetxt(cm_path, cm, delimiter=",", fmt="%d")

    print("\n[OK] Zapisano sanity checki GradCAM:")
    print(" - pred:", out_pred)
    print(" - top2:", out_top2)
    print(" - masked_pred:", out_mask)
    print("[OK] Metryki:", metrics_path)
