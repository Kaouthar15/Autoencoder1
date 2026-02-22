import torch
import torch.nn as nn

class CDAE(nn.Module):
    """
    CDAE = Convolutional Denoising AutoEncoder
    - Convolutional: utilise des convolutions (adapté aux images)
    - Denoising: on apprend à reconstruire une image propre à partir d'une image bruitée
    - AutoEncoder: Encoder -> Latent (bottleneck) -> Decoder -> Reconstruction

    Entrée attendue : x de forme (N, 1, 28, 28), valeurs dans [0, 1]
    Sorties :
      - x_hat : reconstruction (N, 1, 28, 28)
      - z     : représentation latente (feature map) (N, latent_channels, ~4, ~4)
    """
    def __init__(self, latent_channels: int = 16):
        super().__init__()

        # ----------------------------
        # ENCODER
        # ----------------------------
        # Objectif: compresser l'image 28x28 en une représentation plus petite (bottleneck).
        #
        # MaxPool / downsampling:
        # 28x28 -> 14x14 -> 7x7 -> ~4x4
        #
        # Remarque : on garde des convolutions + ReLU pour extraire des motifs (bords, textures, formes)
        self.enc = nn.Sequential(
            # (N,1,28,28) -> (N,32,28,28)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            # (N,32,28,28) -> (N,32,14,14)
            nn.MaxPool2d(kernel_size=2),

            # (N,32,14,14) -> (N,64,14,14)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            # (N,64,14,14) -> (N,64,7,7)
            nn.MaxPool2d(kernel_size=2),

            # (N,64,7,7) -> (N,latent,7,7)
            nn.Conv2d(64, latent_channels, kernel_size=3, padding=1),
            nn.ReLU(),

            # (N,latent,7,7) -> (N,latent,~4,~4)
            # padding=1 ici aide à obtenir ~4x4
            nn.MaxPool2d(kernel_size=2, padding=1),
        )

        # ----------------------------
        # DECODER
        # ----------------------------
        # Objectif: reconstruire l'image depuis le latent z.
        #
        # On remonte la résolution par ConvTranspose2d (upscaling learnable)
        # z (~4x4) -> 8x8 -> 16x16 -> 32x32
        # Puis on "crop" pour revenir à 28x28.
        self.dec = nn.Sequential(
            # (N,latent,4,4) -> (N,64,8,8)
            nn.ConvTranspose2d(latent_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            # (N,64,8,8) -> (N,32,16,16)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            # (N,32,16,16) -> (N,16,32,32)
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            # (N,16,32,32) -> (N,1,32,32)
            # Sigmoid pour sortir dans [0,1] (image normalisée)
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode l'image en latent z."""
        return self.enc(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode le latent z en image reconstruite 28x28."""
        x_hat_32 = self.dec(z)               # (N,1,32,32)
        x_hat_28 = x_hat_32[:, :, 2:30, 2:30]  # crop -> (N,1,28,28)
        return x_hat_28

    def forward(self, x: torch.Tensor):
        """
        Forward complet:
          x -> z -> x_hat
        Retourne reconstruction et latent.
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z