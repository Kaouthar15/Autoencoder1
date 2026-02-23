import torch
import torch.nn as nn


"""Définition du CDAE (Convolutional Denoising AutoEncoder).

l'encodeur est volontairement petit pour que le modèle
ne puisse pas "tout mémoriser" — il doit apprendre ce qui est important pour
la classe normale. Le décodeur remonte la résolution puis on recadre pour obtenir
28×28, ce qui évite de trop se prendre la tête avec des paddings/trous.
"""


class CDAE(nn.Module):
    """
    CDAE = Convolutional Denoising AutoEncoder
    Entrée : (N, 1, 28, 28), valeurs dans [0, 1]
    Sorties :
      - x_hat : reconstruction (N, 1, 28, 28)
      - z     : représentation latente (N, latent_channels, 4, 4)
    """

    def __init__(self, latent_channels: int = 4):
        super().__init__()

        # ----------------------------
        # ENCODER — volontairement simple
        # On garde des couches basiques : conv -> relu -> pool. Le but n'est pas
        # d'avoir un super modèle mais d'imposer une contrainte pour favoriser
        # la spécialisation sur la classe normale.
        # `latent_channels` règle la taille du goulot d'étranglement.
        # ----------------------------
        self.enc = nn.Sequential(
            # (N,1,28,28) -> (N,16,28,28)
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            # (N,16,28,28) -> (N,16,14,14)
            nn.MaxPool2d(kernel_size=2),

            # (N,16,14,14) -> (N,32,14,14)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            # (N,32,14,14) -> (N,32,7,7)
            nn.MaxPool2d(kernel_size=2),

            # (N,32,7,7) -> (N,latent,7,7)
            nn.Conv2d(32, latent_channels, kernel_size=3, padding=1),
            nn.ReLU(),

            # (N,latent,7,7) -> (N,latent,4,4)
            # padding=1 sur le MaxPool pour garder un petit spatial final
            nn.MaxPool2d(kernel_size=2, padding=1),
        )

        # ----------------------------
        # DECODER — on remonte la résolution puis on croppe
        # ConvTranspose2d suffit pour cette taille d'image. On obtient 32×32 puis
        # on coupe les bords pour revenir à 28×28 — c'est un choix pragmatique.
        # ----------------------------
        self.dec = nn.Sequential(
            # (N,latent,4,4) -> (N,32,8,8)
            nn.ConvTranspose2d(latent_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            # (N,32,8,8) -> (N,16,16,16)
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            # (N,16,16,16) -> (N,8,32,32)
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            # (N,8,32,32) -> (N,1,32,32)
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Décoder puis crop pour retrouver 28x28
        x_hat_32 = self.dec(z)                      # (N,1,32,32)
        x_hat_28 = x_hat_32[:, :, 2:30, 2:30]       # crop -> (N,1,28,28)
        return x_hat_28

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z