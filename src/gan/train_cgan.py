import os
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


DATA_ROOT = "data/affectnet_raw/YOLO_format/train"
IMG_DIR = os.path.join(DATA_ROOT, "images")
LBL_DIR = os.path.join(DATA_ROOT, "labels")

NUM_CLASSES = 8
IMG_SIZE = 64
LATENT_DIM = 100
BATCH_SIZE = 128
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        transforms.Grayscale(num_output_channels=1),   # 1 kanał
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),  # zakres [-1,1] dla 1 kanału
    ])


class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_ch=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + num_classes, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, img_ch, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # z: (B, latent_dim), labels: (B,)
        c = self.label_emb(labels)                   # (B, num_classes)
        x = torch.cat([z, c], dim=1)                 # (B, latent_dim + num_classes)
        x = x.view(x.size(0), -1, 1, 1)              # (B, C, 1, 1)
        img = self.net(x)                            # (B, 1, 64, 64)
        return img


class Discriminator(nn.Module):
    def __init__(self, num_classes, img_ch=1):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)

        # lekko słabszy D - mniej kanałów
        self.net = nn.Sequential(
            nn.Conv2d(img_ch + num_classes, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 1, 4, 1, 0, bias=False),
        )

    def forward(self, img, labels):
        # img: (B,1,H,W)
        c = self.label_emb(labels)                   # (B, num_classes)
        c = c.view(c.size(0), c.size(1), 1, 1)       # (B, num_classes, 1, 1)
        c = c.expand(-1, -1, img.size(2), img.size(3))
        x = torch.cat([img, c], dim=1)
        out = self.net(x)                            # (B,1,1,1)
        return out.view(-1)                          # (B,)


def train():
    os.makedirs("models", exist_ok=True)

    dataset = AffectNetGanDataset(
        IMG_DIR,
        LBL_DIR,
        transform=get_train_transform()
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    G = Generator(LATENT_DIM, NUM_CLASSES).to(DEVICE)
    D = Discriminator(NUM_CLASSES).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    noise_std = 0.05  # instance noise

    for epoch in range(EPOCHS):
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for real_imgs, labels in loop:
            real_imgs = real_imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            bsz = real_imgs.size(0)

            # dodaj lekki szum do wejść D
            real_noisy = real_imgs + noise_std * torch.randn_like(real_imgs)
            real_noisy = real_noisy.clamp(-1, 1)

            # update D
            z = torch.randn(bsz, LATENT_DIM, device=DEVICE)
            fake_labels = torch.randint(0, NUM_CLASSES, (bsz,), device=DEVICE)
            fake_imgs = G(z, fake_labels).detach()

            fake_noisy = fake_imgs + noise_std * torch.randn_like(fake_imgs)
            fake_noisy = fake_noisy.clamp(-1, 1)

            D_real = D(real_noisy, labels)
            D_fake = D(fake_noisy, fake_labels)

            real_targets = torch.ones_like(D_real, device=DEVICE)
            fake_targets = torch.zeros_like(D_fake, device=DEVICE)

            loss_D_real = criterion(D_real, real_targets)
            loss_D_fake = criterion(D_fake, fake_targets)
            loss_D = (loss_D_real + loss_D_fake) * 0.5

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # update G
            z = torch.randn(bsz, LATENT_DIM, device=DEVICE)
            gen_labels = torch.randint(0, NUM_CLASSES, (bsz,), device=DEVICE)
            gen_imgs = G(z, gen_labels)

            gen_noisy = gen_imgs + noise_std * torch.randn_like(gen_imgs)
            gen_noisy = gen_noisy.clamp(-1, 1)

            D_gen = D(gen_noisy, gen_labels)
            gen_targets = torch.ones_like(D_gen, device=DEVICE)
            loss_G = criterion(D_gen, gen_targets)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            loop.set_postfix(loss_D=loss_D.item(), loss_G=loss_G.item())

        torch.save(G.state_dict(), "models/gan_generator_cgan.pth")
        print("Zapisano generator po epoce", epoch + 1)


if __name__ == "__main__":
    train()
