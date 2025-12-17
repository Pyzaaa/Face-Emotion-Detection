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
from types import MethodType


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


# ---------------------------
# Attention rollout
# ---------------------------
attn_maps = []

def save_attn_hook(module, inp, out):
    # MultiheadAttention zwraca (attn_output, attn_weights)
    if isinstance(out, tuple) and len(out) >= 2:
        attn_w = out[1]
        if attn_w is None:
            return
        attn_maps.append(attn_w.detach().cpu())
    elif torch.is_tensor(out):
        # awaryjnie, gdyby jakaś wersja zwracała tylko wagi
        attn_maps.append(out.detach().cpu())


def rollout_attention(attn_list):
    if len(attn_list) == 0:
        return None

    result = None
    for attn in attn_list:
        # oczekujemy (B, heads, T, T)
        # jeśli dostaniesz (B, T, T) to dorób heads=1
        if attn.dim() == 3:
            attn = attn.unsqueeze(1)

        attn = attn.mean(dim=1)[0]  # (T, T), B=1
        attn = attn + torch.eye(attn.size(-1))  # residual
        attn = attn / attn.sum(dim=-1, keepdim=True)

        result = attn if result is None else (attn @ result)

    mask = result[0, 1:]  # z CLS do patchy (T-1,)
    return mask


def save_overlay(img_pil, mask_1d, out_path):
    grid = int(np.sqrt(mask_1d.shape[0]))  # dla ViT-B/16: 14
    mask = mask_1d.reshape(grid, grid)

    mask = mask - mask.min()
    mask = mask / (mask.max() + 1e-6)

    mask = cv2.resize(mask, (224, 224))
    mask_uint8 = np.uint8(255 * mask)

    heatmap = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    img = np.array(img_pil.resize((224, 224)))
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    plt.figure(figsize=(4, 4))
    plt.imshow(overlay)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def force_vit_return_attn_weights(model):
    # Wymusza, żeby każde self_attention zwracało attn_weights
    for layer in model.encoder.layers:
        mha = layer.self_attention
        orig_forward = mha.forward

        def forward_with_weights(self, *args, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False  # wtedy jest (B, heads, T, T)
            return orig_forward(*args, **kwargs)

        mha.forward = MethodType(forward_with_weights, mha)


if __name__ == "__main__":
    out_dir = "vit_rollout"
    os.makedirs(out_dir, exist_ok=True)

    test_img_dir = "../../data/affectnet_raw/YOLO_format/test/images"
    test_lbl_dir = "../../data/affectnet_raw/YOLO_format/test/labels"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = AffectNetYOLOTestDataset(test_img_dir, test_lbl_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.vit_b_16(weights=None)
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, 8)
    model.load_state_dict(torch.load("../../models/vit_b16_best.pth", map_location=device))
    model.to(device)
    model.eval()

    force_vit_return_attn_weights(model)

    emotion_classes = ["Anger", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    hooks = []
    for layer in model.encoder.layers:
        hooks.append(layer.self_attention.register_forward_hook(save_attn_hook))

    all_labels, all_preds = [], []

    for imgs, labels, names in tqdm(test_loader, desc="Evaluating ViT + rollout"):
        imgs = imgs.to(device)
        labels = labels.to(device)

        for i in range(len(imgs)):
            attn_maps.clear()

            img = imgs[i].unsqueeze(0)
            label = labels[i].item()
            name = names[i]

            img_path = os.path.join(test_img_dir, name)
            original_pil = Image.open(img_path).convert("RGB")

            with torch.no_grad():
                out = model(img)
                pred = torch.argmax(out).item()

            all_labels.append(label)
            all_preds.append(pred)

            mask_1d = rollout_attention(attn_maps)
            if mask_1d is None:
                continue

            mask_1d = mask_1d.numpy()
            save_path = os.path.join(out_dir, f"{name.split('.')[0]}_rollout.png")
            save_overlay(original_pil, mask_1d, save_path)

    for h in hooks:
        h.remove()

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(8)), average=None, zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    report = classification_report(all_labels, all_preds, target_names=emotion_classes, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(8)))

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
    plt.title("Confusion matrix - ViT rollout")
    plt.colorbar()
    ticks = np.arange(len(emotion_classes))
    plt.xticks(ticks, emotion_classes, rotation=45, ha="right")
    plt.yticks(ticks, emotion_classes)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()

    print("\n[OK] Wyniki + rollout zapisane")
