import os
import torch
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

LATENT_DIM = 100
NUM_CLASSES = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GEN_PATH = "models/gan_cwgan_gp_generator.pth"


class Generator(torch.nn.Module):
    def __init__(self, latent_dim, num_classes, img_ch=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_emb = torch.nn.Embedding(num_classes, num_classes)

        in_dim = latent_dim + num_classes

        self.net = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_dim, 512, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(64, img_ch, 4, 2, 1, bias=False),
            torch.nn.Tanh(),
        )

    def forward(self, z, labels):
        c = self.label_emb(labels) 
        x = torch.cat([z, c], dim=1) 
        x = x.view(x.size(0), -1, 1, 1) 
        img = self.net(x)     
        return img


def denorm(x):
    return (x * 0.5 + 0.5).clamp(0, 1)


def generate_for_class(G, class_id, n_samples=64, nrow=8, epoch_label="final"):
    G.eval()
    os.makedirs("experiments/cwgan_gp_class_samples", exist_ok=True)

    with torch.no_grad():
        z = torch.randn(n_samples, LATENT_DIM, device=DEVICE)
        labels = torch.full((n_samples,), class_id, dtype=torch.long, device=DEVICE)
        imgs = G(z, labels)
        imgs = denorm(imgs.cpu())

        grid = make_grid(imgs, nrow=nrow)
        out_path = f"experiments/cwgan_gp_class_samples/class_{class_id}_{epoch_label}.png"
        save_image(grid, out_path)

    print(f"[OK] zapisano próbki klasy {class_id} do {out_path}")

    plt.figure(figsize=(5, 5))
    plt.imshow(grid.permute(1, 2, 0), cmap="gray")
    plt.axis("off")
    plt.title(f"Klasa {class_id}")
    plt.show()


if __name__ == "__main__":
    G = Generator(LATENT_DIM, NUM_CLASSES).to(DEVICE)
    G.load_state_dict(torch.load(GEN_PATH, map_location=DEVICE))
    G.to(DEVICE)

    print(f"Używam urządzenia: {DEVICE}")
    print("Generuję próbki dla wszystkich klas 0–7...")

    for cls in range(NUM_CLASSES):
        generate_for_class(G, class_id=cls, n_samples=64, nrow=8, epoch_label="final")
