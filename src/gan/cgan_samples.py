import os
import torch
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

from train_cgan import Generator, LATENT_DIM, NUM_CLASSES, IMG_SIZE, DEVICE

OUT_DIR = "experiments/cgan_samples"
os.makedirs(OUT_DIR, exist_ok=True)


def denorm(x):
    return (x * 0.5 + 0.5).clamp(0, 1)


def preview_samples(class_id=0, n_rows=8, n_cols=8, epoch_label="X"):
    """
    Generuje siatkę n_rows x n_cols obrazków GANa dla danej klasy.
    """

    print(f"[INFO] Generuję próbki dla klasy {class_id}...")

    G = Generator(LATENT_DIM, NUM_CLASSES).to(DEVICE)
    G.load_state_dict(torch.load("models/gan_generator_cgan.pth", map_location=DEVICE))
    G.eval()

    n_total = n_rows * n_cols

    with torch.no_grad():
        z = torch.randn(n_total, LATENT_DIM, device=DEVICE)
        labels = torch.full((n_total,), class_id, dtype=torch.long, device=DEVICE)
        fake_imgs = G(z, labels)

    fake_imgs = denorm(fake_imgs.cpu())

    grid = make_grid(fake_imgs, nrow=n_cols)

    out_path = os.path.join(OUT_DIR, f"class_{class_id}_epoch_{epoch_label}.png")
    save_image(grid, out_path)

    print(f"[OK] Zapisano podgląd obrazków do: {out_path}")

    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    preview_samples(class_id=0, epoch_label="test")
