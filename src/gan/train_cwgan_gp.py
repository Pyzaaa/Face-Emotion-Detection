import os
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from PIL import Image
from tqdm import tqdm


DATA_ROOT = "data/affectnet_raw/YOLO_format/train"
IMG_DIR = os.path.join(DATA_ROOT, "images")
LBL_DIR = os.path.join(DATA_ROOT, "labels")

NUM_CLASSES = 8
IMG_SIZE = 64
LATENT_DIM = 100
BATCH_SIZE = 64
EPOCHS = 20

N_CRITIC = 5        # ile kroków krytyka na 1 krok generatora
LAMBDA_GP = 10.0    # współczynnik gradient penalty

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# Dataset
# ---------------------------

class AffectNetGanDataset(Dataset):
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


def get_train_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),   # prostsze zadanie
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),   # zakres [-1,1]
    ])


# ---------------------------
# WGAN-GP: Generator i Krytyk
# ---------------------------

class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_ch=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_emb = nn.Embedding(num_classes, num_classes)

        in_dim = latent_dim + num_classes

        self.net = nn.Sequential(
            # 1x1 -> 4x4
            nn.ConvTranspose2d(in_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, img_ch, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        # z: (B, latent_dim), labels: (B,)
        c = self.label_emb(labels)            # (B, num_classes)
        x = torch.cat([z, c], dim=1)          # (B, latent_dim + num_classes)
        x = x.view(x.size(0), -1, 1, 1)       # (B, in_dim, 1, 1)
        img = self.net(x)                     # (B, 1, 64, 64)
        return img



class Critic(nn.Module):
    def __init__(self, num_classes, img_ch=1):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)

        # część konwolucyjna - bez ostatniej Conv(4x4)
        self.features = nn.Sequential(
            nn.Conv2d(img_ch + num_classes, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # zamiast conv 4x4 robimy pooling + liniową
        self.fc = nn.Linear(512, 1)

    def forward(self, img, labels):
        # img: (B,1,H,W)
        c = self.label_emb(labels)                   # (B, num_classes)
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.expand(-1, -1, img.size(2), img.size(3))
        x = torch.cat([img, c], dim=1)

        feat = self.features(x)                      # (B,512,H',W')
        feat = feat.mean(dim=[2, 3])                 # global average pooling -> (B,512)
        out = self.fc(feat)                          # (B,1)
        return out.view(-1)                          # (B,)

# ---------------------------
# Gradient penalty
# ---------------------------

def gradient_penalty(critic, real_imgs, fake_imgs, labels):
    bsz = real_imgs.size(0)
    alpha = torch.rand(bsz, 1, 1, 1, device=DEVICE)
    interpolates = alpha * real_imgs + (1 - alpha) * fake_imgs
    interpolates.requires_grad_(True)

    d_interpolates = critic(interpolates, labels)

    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates, device=DEVICE),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(bsz, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def denorm(x):
    # z [-1,1] na [0,1]
    return (x * 0.5 + 0.5).clamp(0, 1)


# ---------------------------
# Trening
# ---------------------------

def train():
    os.makedirs("models", exist_ok=True)
    os.makedirs("experiments/gan_samples", exist_ok=True)

    dataset = AffectNetGanDataset(
        IMG_DIR,
        LBL_DIR,
        transform=get_train_transform()
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    G = Generator(LATENT_DIM, NUM_CLASSES).to(DEVICE)
    C = Critic(NUM_CLASSES).to(DEVICE)

    opt_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_C = torch.optim.Adam(C.parameters(), lr=1e-4, betas=(0.0, 0.9))

    step = 0

    for epoch in range(EPOCHS):
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for real_imgs, labels in loop:
            real_imgs = real_imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            bsz = real_imgs.size(0)

            # trenuj krytyka N_CRITIC razy
            for _ in range(N_CRITIC):
                z = torch.randn(bsz, LATENT_DIM, device=DEVICE)
                fake_imgs = G(z, labels).detach()

                crit_real = C(real_imgs, labels)
                crit_fake = C(fake_imgs, labels)

                gp = gradient_penalty(C, real_imgs, fake_imgs, labels)
                loss_C = -(crit_real.mean() - crit_fake.mean()) + LAMBDA_GP * gp

                opt_C.zero_grad()
                loss_C.backward()
                opt_C.step()

            # trenuj generator
            z = torch.randn(bsz, LATENT_DIM, device=DEVICE)
            gen_imgs = G(z, labels)
            crit_fake_for_g = C(gen_imgs, labels)
            loss_G = -crit_fake_for_g.mean()

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            loop.set_postfix(loss_C=loss_C.item(), loss_G=loss_G.item())
            step += 1

            # co pewien czas zapisujemy próbki
            if step % 200 == 0:
                G.eval()
                with torch.no_grad():
                    n_samples = 64
                    z_sample = torch.randn(n_samples, LATENT_DIM, device=DEVICE)
                    # np. wszystkie próbki z klasy 0
                    labels_sample = torch.zeros(n_samples, dtype=torch.long, device=DEVICE)
                    fake_sample = G(z_sample, labels_sample)
                    fake_sample = denorm(fake_sample.cpu())
                    grid = make_grid(fake_sample, nrow=8)
                    out_path = f"experiments/gan_samples/epoch{epoch+1}_step{step}_class0.png"
                    save_image(grid, out_path)
                G.train()

        torch.save(G.state_dict(), "models/gan_cwgan_gp_generator.pth")
        print(f"[OK] Zapisano generator po epoce {epoch+1}")

    print("Trening WGAN-GP zakończony.")


if __name__ == "__main__":
    train()
