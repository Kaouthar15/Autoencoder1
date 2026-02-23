import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from app.model import CDAE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def add_noise(x: torch.Tensor, noise: float = 0.3) -> torch.Tensor:
    return (x + noise * torch.randn_like(x)).clamp(0.0, 1.0)

def percentile(values, p: float):
    values = sorted(values)
    if not values:
        return 0.0
    idx = int(p * (len(values) - 1))
    return values[idx]

def main():

    # ----------------------------
    # PARAMÈTRES
    # ----------------------------
    normal_class = 9          # ✅ chiffre 0 = classe normale
    latent_channels = 2
    noise = 0.3
    epochs = 20               # ⬅️ réduit pour gagner du temps
    batch_size = 256
    lr = 1e-3
    weight_decay = 1e-4

    # ----------------------------
    # DATASET: MNIST (PLUS STABLE)
    # ----------------------------
    tfm = transforms.ToTensor()

    train_ds = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=tfm
    )

    test_ds = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=tfm
    )

    # 🔥 On entraîne uniquement sur le chiffre 0
    train_idx = [i for i, (_, y) in enumerate(train_ds) if y == normal_class]
    train_norm = Subset(train_ds, train_idx)

    train_loader = DataLoader(train_norm, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # ----------------------------
    # MODEL
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
            x = x.to(DEVICE)
            x_noisy = add_noise(x, noise)

            x_hat, _ = model(x_noisy)
            loss = loss_fn(x_hat, x)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)
            n += x.size(0)

        print(f"Epoch {ep}/{epochs} - loss: {total_loss / max(n,1):.6f}")

    # ----------------------------
    # CALCUL THRESHOLD (p98)
    # ----------------------------
    model.eval()
    normal_scores = []

    with torch.no_grad():
        for x, _ in train_loader:
            x = x.to(DEVICE)
            x_noisy = add_noise(x, noise)
            x_hat, _ = model(x_noisy)

            mse = ((x - x_hat) ** 2).mean(dim=(1,2,3))
            normal_scores.extend(mse.cpu().tolist())

    threshold = percentile(normal_scores, 0.98)
    print("Threshold (p98 train normal):", threshold)

    # ----------------------------
    # SAVE
    # ----------------------------
    assets_dir = os.path.join("app", "assets")
    os.makedirs(assets_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(assets_dir, "cdae.pt"))

    with open(os.path.join(assets_dir, "threshold.json"), "w") as f:
        json.dump(
            {
                "threshold": threshold,
                "normal_class": normal_class,
                "noise": noise,
                "latent_channels": latent_channels
            },
            f,
            indent=2
        )

    print("Saved model and threshold.")

    # ----------------------------
    # TEST CHECK (0 vs 3)
    # ----------------------------
    test_scores_0 = []
    test_scores_3 = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            x_noisy = add_noise(x, noise)
            x_hat, _ = model(x_noisy)

            mse = ((x - x_hat) ** 2).mean(dim=(1,2,3)).cpu().tolist()

            for s, yy in zip(mse, y.tolist()):
                if yy == 0:
                    test_scores_0.append(float(s))
                elif yy == 3:
                    test_scores_3.append(float(s))

    print("Test class 9 mean:", sum(test_scores_0)/len(test_scores_0))
    print("Test class 3 mean:", sum(test_scores_3)/len(test_scores_3))


if __name__ == "__main__":
    main()