import os
import shutil
from collections import Counter

import torch
from torchvision.utils import save_image


# ścieżki
RAW_ROOT = "data/affectnet_raw/YOLO_format"
RAW_TRAIN_IMG = os.path.join(RAW_ROOT, "train", "images")
RAW_TRAIN_LBL = os.path.join(RAW_ROOT, "train", "labels")

AUG_ROOT = "data/affectnet_cwgan_aug/YOLO_format"
AUG_TRAIN_IMG = os.path.join(AUG_ROOT, "train", "images")
AUG_TRAIN_LBL = os.path.join(AUG_ROOT, "train", "labels")

os.makedirs(AUG_TRAIN_IMG, exist_ok=True)
os.makedirs(AUG_TRAIN_LBL, exist_ok=True)

# parametry modelu - takie jak w train_cwgan_gp
LATENT_DIM = 100
NUM_CLASSES = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GEN_PATH = "models/gan_cwgan_gp_generator.pth"


# ---------------------------
# Generator - ten sam jak w treningu
# ---------------------------
class Generator(torch.nn.Module):
    def __init__(self, latent_dim, num_classes, img_ch=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_emb = torch.nn.Embedding(num_classes, num_classes)

        in_dim = latent_dim + num_classes

        self.net = torch.nn.Sequential(
            # 1x1 -> 4x4
            torch.nn.ConvTranspose2d(in_dim, 512, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(True),

            # 4x4 -> 8x8
            torch.nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),

            # 8x8 -> 16x16
            torch.nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),

            # 16x16 -> 32x32
            torch.nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True),

            # 32x32 -> 64x64
            torch.nn.ConvTranspose2d(64, img_ch, 4, 2, 1, bias=False),
            torch.nn.Tanh(),
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)            # (B, num_classes)
        x = torch.cat([z, c], dim=1)          # (B, latent_dim+num_classes)
        x = x.view(x.size(0), -1, 1, 1)       # (B, in_dim, 1, 1)
        img = self.net(x)                     # (B, 1, 64, 64)
        return img


def denorm(x):
    # z [-1,1] do [0,1]
    return (x * 0.5 + 0.5).clamp(0, 1)


# ---------------------------
# Liczenie klas w oryginalnym train
# ---------------------------
def count_classes(lbl_dir):
    counts = Counter()
    for fname in os.listdir(lbl_dir):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(lbl_dir, fname)
        with open(path, "r") as f:
            line = f.readline().strip()
            if not line:
                continue
            cls = int(line.split()[0])
            counts[cls] += 1
    return counts


# ---------------------------
# Kopiowanie oryginalnych danych do nowego train
# ---------------------------
def copy_original_train():
    print("[INFO] Kopiuję oryginalny train do affectnet_cwgan_aug...")
    for fname in os.listdir(RAW_TRAIN_IMG):
        src = os.path.join(RAW_TRAIN_IMG, fname)
        dst = os.path.join(AUG_TRAIN_IMG, fname)
        shutil.copy2(src, dst)

    for fname in os.listdir(RAW_TRAIN_LBL):
        src = os.path.join(RAW_TRAIN_LBL, fname)
        dst = os.path.join(AUG_TRAIN_LBL, fname)
        shutil.copy2(src, dst)
    print("[OK] Skopiowano oryginalny train.")


# ---------------------------
# Generowanie brakujących próbek
# ---------------------------
def generate_missing_samples():
    # policz klasy na oryginalnym zbiorze
    counts = count_classes(RAW_TRAIN_LBL)
    print("[INFO] Liczności klas w oryginalnym train:")
    for c in range(NUM_CLASSES):
        print(f"  klasa {c}: {counts.get(c, 0)}")

    target = max(counts.values())
    print(f"[INFO] Cel na klasę (target) = {target}")

    # wczytaj generator
    G = Generator(LATENT_DIM, NUM_CLASSES).to(DEVICE)
    G.load_state_dict(torch.load(GEN_PATH, map_location=DEVICE))
    G.eval()

    # licznik do nazw plików
    per_class_idx = {c: 0 for c in range(NUM_CLASSES)}

    # znajdź maksymalny istniejący indeks, żeby nie nadpisać
    for fname in os.listdir(AUG_TRAIN_IMG):
        if fname.startswith("gan_cwgan_") and fname.endswith(".png"):
            # format: gan_cwgan_c{cls}_i{idx}.png
            try:
                base = fname.replace(".png", "")
                parts = base.split("_")
                c_part = [p for p in parts if p.startswith("c")]
                i_part = [p for p in parts if p.startswith("i")]
                if c_part and i_part:
                    c = int(c_part[0][1:])
                    i = int(i_part[0][1:])
                    per_class_idx[c] = max(per_class_idx[c], i + 1)
            except Exception:
                continue

    with torch.no_grad():
        for cls in range(NUM_CLASSES):
            current = counts.get(cls, 0)
            need = target - current
            if need <= 0:
                print(f"[INFO] Klasa {cls} już ma {current} próbek, nic nie generuję.")
                continue

            print(f"[INFO] Generuję {need} próbek dla klasy {cls}...")

            remaining = need
            while remaining > 0:
                batch_size = min(64, remaining)
                z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
                labels = torch.full((batch_size,), cls, dtype=torch.long, device=DEVICE)

                fake_imgs = G(z, labels)          # (B,1,64,64)
                fake_imgs = denorm(fake_imgs.cpu())

                for i in range(batch_size):
                    img_tensor = fake_imgs[i]
                    idx = per_class_idx[cls]
                    img_name = f"gan_cwgan_c{cls}_i{idx}.png"
                    lbl_name = f"gan_cwgan_c{cls}_i{idx}.txt"

                    img_path = os.path.join(AUG_TRAIN_IMG, img_name)
                    lbl_path = os.path.join(AUG_TRAIN_LBL, lbl_name)

                    # zapis obrazka
                    save_image(img_tensor, img_path)

                    # YOLO label: klasa + bbox na cały obraz
                    with open(lbl_path, "w") as f:
                        f.write(f"{cls} 0.5 0.5 1.0 1.0\n")

                    per_class_idx[cls] += 1
                    remaining -= 1
                    if remaining <= 0:
                        break

            print(f"[OK] Wygenerowano {need} próbek dla klasy {cls}.")

    # podsumowanie nowego zbioru
    new_counts = count_classes(AUG_TRAIN_LBL)
    print("\n[INFO] Liczności klas w ZBALANSOWANYM train (affectnet_cwgan_aug):")
    for c in range(NUM_CLASSES):
        print(f"  klasa {c}: {new_counts.get(c, 0)}")


if __name__ == "__main__":
    print("[STEP 1] Kopiowanie oryginalnego train...")
    copy_original_train()

    print("\n[STEP 2] Generowanie brakujących próbek cWGAN-GP...")
    generate_missing_samples()

    print("\n[DONE] Zbalansowany zbiór train zapisany w:")
    print(f"  {AUG_TRAIN_IMG}")
    print(f"  {AUG_TRAIN_LBL}")
