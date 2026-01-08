import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import cv2

# -----------------------
# Dataset
# -----------------------
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
        name = self.samples[idx]
        img_path = os.path.join(self.img_dir, name)
        lbl_path = os.path.join(self.lbl_dir, name.rsplit(".", 1)[0] + ".txt")

        img = Image.open(img_path).convert("RGB")
        with open(lbl_path, "r") as f:
            label = int(f.readline().split()[0])

        x = self.transform(img) if self.transform else img
        return x, label, name, img_path

# -----------------------
# GradCAM
# -----------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_acts)
        target_layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, module, inp, out):
        self.activations = out.detach()  # (B,C,H,W)

    def _save_grads(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()  # (B,C,H,W)

    def cam(self):
        # zakładamy B=1
        grads = self.gradients[0]      # (C,H,W)
        acts = self.activations[0]     # (C,H,W)
        w = grads.mean(dim=(1, 2))     # (C,)
        cam = (w[:, None, None] * acts).sum(dim=0)  # (H,W)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-6)
        return cam.cpu().numpy()

# -----------------------
# Maskowanie
# -----------------------
def make_cam_mask(cam_2d, img_hw=(224, 224), topk_frac=0.20):
    """
    topk_frac = jaki % pikseli zasłonić (np. 0.2 = 20% najbardziej aktywnych)
    Zwraca maskę bool (H,W): True = zasłonić.
    """
    cam = cv2.resize(cam_2d, img_hw[::-1])  # (W,H) w OpenCV
    flat = cam.flatten()
    k = int(len(flat) * topk_frac)
    if k < 1:
        k = 1
    thr = np.partition(flat, -k)[-k]  # próg dla top-k
    mask = cam >= thr
    return mask

def make_random_block_mask(img_hw=(224, 224), area_frac=0.20):
    """
    Losowy prostokąt o polu zbliżonym do area_frac * H*W
    """
    H, W = img_hw
    area = int(H * W * area_frac)
    area = max(area, 1)

    # dobieramy wymiary prostokąta losowo
    h = int(np.sqrt(area))
    w = int(area / max(h, 1))
    h = max(1, min(h, H))
    w = max(1, min(w, W))

    y0 = random.randint(0, H - h)
    x0 = random.randint(0, W - w)

    mask = np.zeros((H, W), dtype=bool)
    mask[y0:y0+h, x0:x0+w] = True
    return mask

def apply_mask_tensor(x, mask_bool, fill_value=0.0):
    """
    x: Tensor (3,224,224) po normalizacji ImageNet
    mask_bool: (224,224) True = maskuj
    fill_value w przestrzeni znormalizowanej.
    """
    x2 = x.clone()
    m = torch.from_numpy(mask_bool).to(x2.device)
    x2[:, m] = fill_value
    return x2

def save_debug_triplet(img_path, cam_mask, rnd_mask, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)

    img = np.array(Image.open(img_path).convert("RGB").resize((224, 224)))

    def overlay_mask(img_np, mask_bool, color=(255, 0, 0), alpha=0.45):
        over = img_np.copy()
        over[mask_bool] = (alpha * np.array(color) + (1 - alpha) * over[mask_bool]).astype(np.uint8)
        return over

    cam_over = overlay_mask(img, cam_mask, color=(255, 0, 0), alpha=0.45)
    rnd_over = overlay_mask(img, rnd_mask, color=(0, 255, 0), alpha=0.45)

    Image.fromarray(img).save(os.path.join(out_dir, f"{name}_orig.png"))
    Image.fromarray(cam_over).save(os.path.join(out_dir, f"{name}_cam_mask.png"))
    Image.fromarray(rnd_over).save(os.path.join(out_dir, f"{name}_rnd_mask.png"))

# -----------------------
# MAIN
# -----------------------
def main():
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    test_img_dir = "data/affectnet_raw/YOLO_format/test/images"
    test_lbl_dir = "data/affectnet_raw/YOLO_format/test/labels"

    results_dir = "experiments/heatmap_analysis/resnet_validate"
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, "summary.txt")
    debug_dir = os.path.join(results_dir, "debug_examples")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    ds = AffectNetYOLOTestDataset(test_img_dir, test_lbl_dir, transform)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 8)
    model.load_state_dict(torch.load("models/resnet50_best.pth", map_location=device))
    model.to(device)
    model.eval()

    target_layer = model.layer4[-1]
    cam = GradCAM(model, target_layer)

    # metryki
    n = 0
    correct_orig = 0
    correct_cam = 0
    correct_rnd = 0

    drop_true_cam = []
    drop_true_rnd = []
    drop_pred_cam = []
    drop_pred_rnd = []

    TOPK_FRAC = 0.20      # ile zasłaniamy wg heatmapy
    DEBUG_SAVE = 30       # zapisuj pierwsze N przykładów

    softmax = nn.Softmax(dim=1)

    for x, y, name, img_path in tqdm(loader, desc="Validating heatmaps"):
        x = x.to(device)        # (1,3,224,224)
        y = torch.tensor([y]).to(device)

        # ---------- ORYGINAŁ ----------
        model.zero_grad(set_to_none=True)
        out = model(x)
        prob = softmax(out)

        pred = int(torch.argmax(prob, dim=1).item())
        true = int(y.item())

        p_true = float(prob[0, true].item())
        p_pred = float(prob[0, pred].item())

        correct_orig += int(pred == true)

        # ---------- CAM mask ----------
        # grad pod predykcję (jak w typowym GradCAM)
        model.zero_grad(set_to_none=True)
        out2 = model(x)
        out2[0, pred].backward()
        cam_map = cam.cam()

        cam_mask = make_cam_mask(cam_map, img_hw=(224,224), topk_frac=TOPK_FRAC)
        x_cam = apply_mask_tensor(x[0], cam_mask, fill_value=0.0).unsqueeze(0)

        with torch.no_grad():
            out_cam = model(x_cam)
            prob_cam = softmax(out_cam)
        pred_cam = int(torch.argmax(prob_cam, dim=1).item())
        correct_cam += int(pred_cam == true)

        p_true_cam = float(prob_cam[0, true].item())
        p_pred_cam = float(prob_cam[0, pred].item())

        drop_true_cam.append(p_true - p_true_cam)
        drop_pred_cam.append(p_pred - p_pred_cam)

        # ---------- RANDOM mask ----------
        rnd_mask = make_random_block_mask(img_hw=(224,224), area_frac=TOPK_FRAC)
        x_rnd = apply_mask_tensor(x[0], rnd_mask, fill_value=0.0).unsqueeze(0)

        with torch.no_grad():
            out_rnd = model(x_rnd)
            prob_rnd = softmax(out_rnd)
        pred_rnd = int(torch.argmax(prob_rnd, dim=1).item())
        correct_rnd += int(pred_rnd == true)

        p_true_rnd = float(prob_rnd[0, true].item())
        p_pred_rnd = float(prob_rnd[0, pred].item())

        drop_true_rnd.append(p_true - p_true_rnd)
        drop_pred_rnd.append(p_pred - p_pred_rnd)

        # debug przykłady
        if n < DEBUG_SAVE:
            save_debug_triplet(img_path[0], cam_mask, rnd_mask, debug_dir, name[0].rsplit(".", 1)[0])

        n += 1

    acc_orig = correct_orig / n
    acc_cam = correct_cam / n
    acc_rnd = correct_rnd / n

    def mean(x):
        return float(np.mean(x)) if len(x) else 0.0

    lines = []
    lines.append("ResNet50 heatmap validation (GradCAM)")
    lines.append(f"samples={n}")
    lines.append(f"mask_topk_frac={TOPK_FRAC}")
    lines.append("")
    lines.append(f"acc_original={acc_orig:.4f}")
    lines.append(f"acc_mask_cam={acc_cam:.4f}")
    lines.append(f"acc_mask_random={acc_rnd:.4f}")
    lines.append("")
    lines.append("avg probability drop (higher is better, means mask hits important region)")
    lines.append(f"drop_true_cam={mean(drop_true_cam):.4f}")
    lines.append(f"drop_true_random={mean(drop_true_rnd):.4f}")
    lines.append(f"drop_pred_cam={mean(drop_pred_cam):.4f}")
    lines.append(f"drop_pred_random={mean(drop_pred_rnd):.4f}")
    lines.append("")
    lines.append(f"debug_examples_dir={debug_dir}")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"[OK] saved: {summary_path}")

if __name__ == "__main__":
    main()
