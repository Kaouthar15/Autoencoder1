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
    latent_channels = 4       # bottleneck channels (plus petit = plus discriminant)
    noise = 0.3               # bruit utilisé pour denoising (train + calibration threshold)
    epochs = 40
    batch_size = 256
    lr = 1e-3

    # Régularisation légère (réduit la généralisation)
    weight_decay = 1e-5

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
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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
    # THRESHOLD CALIBRÉ COMME L'INFÉRENCE (noisy->clean)
    # ----------------------------
    model.eval()
    normal_scores = []

    with torch.no_grad():
        for x, _ in train_loader:
            x = x.to(DEVICE)               # clean reference
            x_noisy = add_noise(x, noise)  # noisy input (même bruit)
            x_hat, _ = model(x_noisy)

            # Score par image = MSE(x_clean, x_hat)
            mse_per_img = ((x - x_hat) ** 2).mean(dim=(1, 2, 3)).detach().cpu().tolist()
            normal_scores.extend([float(s) for s in mse_per_img])

    # p98 = moins de faux "anomaly" sur des vrais t-shirts
    threshold = percentile(normal_scores, 0.98)
    print("Threshold (p98 train normal, noisy->clean):", threshold)

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
                "weight_decay": weight_decay,
            },
            f,
            indent=2
        )

    print("Saved:", os.path.join(assets_dir, "cdae.pt"), "and threshold.json")

    # ----------------------------
    # (Optionnel) CHECK GLOBAL: classe 0 vs toutes les autres classes (1..9)
    # Objectif : ID = classe 0, OOD/anomaly = classes 1..9
    # ----------------------------
    model.eval()
    scores_normal = []
    scores_anom = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            y_cpu = y.cpu()

            x_noisy = add_noise(x, noise)
            x_hat, _ = model(x_noisy)

            mse = ((x - x_hat) ** 2).mean(dim=(1, 2, 3)).detach().cpu().tolist()
            for s, yy in zip(mse, y_cpu.tolist()):
                if yy == normal_class:
                    scores_normal.append(float(s))
                else:
                    scores_anom.append(float(s))

    if scores_normal:
        mean_n = sum(scores_normal) / len(scores_normal)
        fpr = sum(1 for s in scores_normal if s > threshold) / len(scores_normal)
        print(f"Test NORMAL (class {normal_class}) mean score: {mean_n:.6f}")
        print(f"FPR (normal flagged anomaly): {100*fpr:.2f}%")

    if scores_anom:
        mean_a = sum(scores_anom) / len(scores_anom)
        tpr = sum(1 for s in scores_anom if s > threshold) / len(scores_anom)
        print(f"Test ANOMALY (classes != {normal_class}) mean score: {mean_a:.6f}")
        print(f"TPR (anomaly detected): {100*tpr:.2f}%")


if __name__ == "__main__":
    main()