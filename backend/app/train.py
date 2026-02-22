import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from app.model import CDAE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def add_noise(x: torch.Tensor, noise: float = 0.3) -> torch.Tensor:
    """Ajoute du bruit gaussien et clip dans [0,1]."""
    return (x + noise * torch.randn_like(x)).clamp(0.0, 1.0)

def percentile(values, p: float):
    """Percentile simple (p en [0,1])."""
    values = sorted(values)
    if not values:
        return 0.0
    idx = int(p * (len(values) - 1))
    return values[idx]

def main():
    # ----------------------------
    # PARAMÈTRES
    # ----------------------------
    normal_class = 0          # 0 = T-shirt/top (Fashion-MNIST)
    latent_channels = 16      # bottleneck channels
    noise = 0.3               # bruit utilisé pour denoising (train + calibration threshold)
    epochs = 40               # tu peux garder 40
    batch_size = 256
    lr = 1e-3

    # ----------------------------
    # DATASET: Fashion-MNIST
    # ----------------------------
    tfm = transforms.ToTensor()
    train_ds = datasets.FashionMNIST(root="data", train=True, download=True, transform=tfm)
    test_ds  = datasets.FashionMNIST(root="data", train=False, download=True, transform=tfm)

    # On entraîne seulement sur la classe "normale"
    train_idx = [i for i, (_, y) in enumerate(train_ds) if y == normal_class]
    train_norm = Subset(train_ds, train_idx)

    # DataLoaders
    train_loader = DataLoader(train_norm, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # ----------------------------
    # MODEL + OPT + LOSS
    # ----------------------------
    model = CDAE(latent_channels=latent_channels).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # ----------------------------
    # TRAINING (noisy -> clean)
    # ----------------------------
    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        n = 0

        for x, _ in train_loader:
            x = x.to(DEVICE)              # clean target
            x_noisy = add_noise(x, noise) # noisy input

            x_hat, _ = model(x_noisy)
            loss = loss_fn(x_hat, x)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)
            n += x.size(0)

        print(f"Epoch {ep}/{epochs} - loss: {total_loss / max(n,1):.6f}")

    # ----------------------------
    # ✅ SOLUTION UNIQUE: THRESHOLD CALIBRÉ COMME L'INFÉRENCE
    #
    # On calcule le threshold sur les "normaux" du TRAIN, en condition noisy->clean,
    # comme quand le backend fait: x_noisy -> x_hat puis compare à x_clean.
    # ----------------------------
    model.eval()
    normal_scores = []

    with torch.no_grad():
        for x, _ in train_loader:
            x = x.to(DEVICE)               # clean reference
            x_noisy = add_noise(x, noise)  # noisy input (même bruit)
            x_hat, _ = model(x_noisy)

            # Score par image = MSE(x_clean, x_hat)
            mse_per_img = ((x - x_hat) ** 2).mean(dim=(1,2,3)).detach().cpu().tolist()
            normal_scores.extend([float(s) for s in mse_per_img])

    threshold = percentile(normal_scores, 0.95)
    print("Threshold (p95 train normal, noisy->clean):", threshold)

    # ----------------------------
    # SAVE: weights + threshold config
    # ----------------------------
    assets_dir = os.path.join("app", "assets")
    os.makedirs(assets_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(assets_dir, "cdae.pt"))

    with open(os.path.join(assets_dir, "threshold.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "threshold": threshold,
                "normal_class": normal_class,
                "noise": noise,
                "latent_channels": latent_channels,
            },
            f,
            indent=2
        )

    print("Saved:", os.path.join(assets_dir, "cdae.pt"), "and threshold.json")

    # (Optionnel) mini check sur le test set: juste afficher moyenne score sur test normaux
    # Ça aide à voir si on est “dans les clous”
    model.eval()
    test_scores_norm = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            y = y.cpu()

            x_noisy = add_noise(x, noise)
            x_hat, _ = model(x_noisy)

            mse = ((x - x_hat) ** 2).mean(dim=(1,2,3)).detach().cpu().tolist()
            for s, yy in zip(mse, y.tolist()):
                if yy == normal_class:
                    test_scores_norm.append(float(s))

    if test_scores_norm:
        print("Test normal mean score:", sum(test_scores_norm) / len(test_scores_norm))

if __name__ == "__main__":
    main()