"""
on entraîne un petit autoencodeur uniquement sur une classe "normale" pour qu'il reconstruise bien
cette classe et se trompe sur tout le reste. On sauve ensuite un seuil (percentile)
qui sert à décider si une image est anormale.

Pourquoi faire du "denoising" (noisy -> clean) ? Parce que si on ajoute du bruit
à l'entrée pendant l'entraînement, le réseau apprend à extraire les caractéristiques
essentielles et n'apprend pas à mémoriser les détails.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from app.model import CDAE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def add_noise(x: torch.Tensor, noise: float = 0.5) -> torch.Tensor:
    """Ajoute du bruit gaussien aux images (utilisé seulement à l'entraînement).

    on utilise ce bruit pour rendre la tâche plus robuste. Le réseau
    apprend à ignorer le "bruit" et à garder l'information utile.
    """
    return (x + noise * torch.randn_like(x)).clamp(0.0, 1.0)


def percentile(values, p: float):
    """Renvoie le percentile p d'une liste (ex: p=0.95 -> 95%).

    Pourquoi on fait ça ? Parce que pour fixer un seuil de détection on préfère
    une position dans la distribution (percentile) plutôt qu'une moyenne, ça
    évite d'être trop affecté par des valeurs parasites.
    """
    values = sorted(values)
    if not values:
        return 0.0
    idx = int(p * (len(values) - 1))
    return values[idx]


def main():

    # ----------------------------
    # PARAMÈTRES (modifiables facilement)
    # ----------------------------
    # La classe que l'on considère comme "normale" — on n'entraîne que dessus.
    normal_class    = 0

    # Taille du canal latent. Plus petit = modèle plus contraint, donc plus
    # porté à spécialiser sa reconstruction.
    latent_channels = 4

    # Force du bruit lors de l'entraînement (noisy -> clean)
    noise           = 0.5

    # Optimisation / durée
    epochs          = 30
    batch_size      = 256
    lr              = 1e-3
    weight_decay    = 1e-4

    # ----------------------------
    # DATASET : MNIST
    # ----------------------------
    tfm = transforms.ToTensor()

    train_ds = datasets.MNIST(root="data", train=True,  download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="data", train=False, download=True, transform=tfm)

    # ✅ Entraînement UNIQUEMENT sur la classe normale
    train_idx  = [i for i, (_, y) in enumerate(train_ds) if y == normal_class]
    train_norm = Subset(train_ds, train_idx)

    train_loader = DataLoader(train_norm, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,   batch_size=batch_size, shuffle=False)

    # ----------------------------
    # MODEL
    # ----------------------------
    model   = CDAE(latent_channels=latent_channels).to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    # ----------------------------
    # TRAINING  (noisy -> clean)
    # ----------------------------
    model.train()

    for ep in range(1, epochs + 1):
        total_loss = 0.0
        n = 0

        for x, _ in train_loader:
            x       = x.to(DEVICE)
            x_noisy = add_noise(x, noise)

            x_hat, _ = model(x_noisy)
            loss      = loss_fn(x_hat, x)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)
            n          += x.size(0)

        print(f"Epoch {ep}/{epochs} - loss: {total_loss / max(n, 1):.6f}")

    # ----------------------------
    # THRESHOLD  (p95 sur images PROPRES)
    # ----------------------------
    # Calcul du seuil sur les reconstructions des images propres (sans bruit).
    # Remarques:
    # - On utilise un percentile (ici p95) pour fixer un seuil robuste.
    # - Le choix de p95 vs p98/90 dépend du compromis entre faux positifs/négatifs.
    # - Important: l'inférence attend des images propres; c'est cohérent avec
    #   le calcul du seuil ci-dessous.
    # ----------------------------
    model.eval()
    normal_scores = []

    with torch.no_grad():
        for x, _ in train_loader:
            x = x.to(DEVICE)
            x_hat, _ = model(x)   # image propre en entrée

            mse = ((x - x_hat) ** 2).mean(dim=(1, 2, 3))
            normal_scores.extend(mse.cpu().tolist())

    threshold = percentile(normal_scores, 0.90)
    print(f"\nThreshold (p90 train normal class={normal_class}): {threshold:.6f}")

    # ----------------------------
    # SAVE
    # ----------------------------
    assets_dir = os.path.join("app", "assets")
    os.makedirs(assets_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(assets_dir, "cdae.pt"))

    with open(os.path.join(assets_dir, "threshold.json"), "w") as f:
        json.dump(
            {
                "threshold":       threshold,
                "normal_class":    normal_class,
                "noise":           noise,
                "latent_channels": latent_channels,
            },
            f,
            indent=2,
        )

    print("Saved model and threshold.")

    # ----------------------------
    # TEST CHECK  (classe normale vs anomalie)
    # ----------------------------
    anomaly_class = 3

    scores_normal  = []
    scores_anomaly = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            x_hat, _ = model(x)   # pas de bruit en inférence

            mse = ((x - x_hat) ** 2).mean(dim=(1, 2, 3)).cpu().tolist()

            for s, yy in zip(mse, y.tolist()):
                if yy == normal_class:
                    scores_normal.append(float(s))
                elif yy == anomaly_class:
                    scores_anomaly.append(float(s))

    mean_normal  = sum(scores_normal)  / max(len(scores_normal),  1)
    mean_anomaly = sum(scores_anomaly) / max(len(scores_anomaly), 1)

    print(f"\nTest class {normal_class}  (normal)  mean MSE : {mean_normal:.6f}")
    print(f"Test class {anomaly_class} (anomaly) mean MSE : {mean_anomaly:.6f}")
    print(f"Threshold                            : {threshold:.6f}")
    print(f"→ Ratio anomaly/normal : {mean_anomaly / max(mean_normal, 1e-9):.2f}x  (idéal > 2x)")

    # ----------------------------
    # ACCURACY GLOBALE
    # ----------------------------
    correct = 0
    total   = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            x_hat, _ = model(x)

            mse = ((x - x_hat) ** 2).mean(dim=(1, 2, 3)).cpu().tolist()

            for s, yy in zip(mse, y.tolist()):
                is_anomaly_pred = s > threshold
                is_anomaly_true = (yy != normal_class)
                if is_anomaly_pred == is_anomaly_true:
                    correct += 1
                total += 1

    print(f"\nAccuracy détection anomalie (tout le test set) : {correct / total * 100:.2f}%")


if __name__ == "__main__":
    main()