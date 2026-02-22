import os
import json
import io
import base64
import numpy as np
import torch
from PIL import Image

from app.model import CDAE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# On garde le modèle en mémoire pour éviter de le recharger à chaque requête
_model = None
_cfg = None

def load_artifacts():
    """
    Charge 1 seule fois:
    - le modèle (poids cdae.pt)
    - la config threshold.json (threshold, latent_channels, etc.)
    """
    global _model, _cfg

    if _model is not None and _cfg is not None:
        return _model, _cfg

    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    weights_path = os.path.join(assets_dir, "cdae.pt")
    cfg_path = os.path.join(assets_dir, "threshold.json")

    if not os.path.exists(weights_path) or not os.path.exists(cfg_path):
        raise FileNotFoundError(
            "Poids/config manquants. Lance d'abord l'entraînement: .venv\\Scripts\\python -m app.train"
        )

    with open(cfg_path, "r", encoding="utf-8") as f:
        _cfg = json.load(f)

    # Créer le modèle avec le même latent_channels que lors du training
    _model = CDAE(latent_channels=int(_cfg["latent_channels"])).to(DEVICE)

    # Charger les poids
    state = torch.load(weights_path, map_location=DEVICE)
    _model.load_state_dict(state)
    _model.eval()

    return _model, _cfg

def pil_to_base64(img: Image.Image) -> str:
    """Convertit une image PIL -> string base64 PNG (pour l'envoyer au front)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    """
    Prépare une image uploadée:
    - grayscale
    - resize 28x28
    - normalise [0,1]
    - shape (1,1,28,28)
    """
    img = pil_img.convert("L").resize((28, 28))
    arr = np.array(img).astype(np.float32) / 255.0
    x = torch.from_numpy(arr)[None, None, :, :]  # (1,1,28,28)
    return x

def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """Tensor (1,1,28,28) -> PIL grayscale 28x28."""
    arr = (x.squeeze().detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")

def make_heatmap(err_map: np.ndarray) -> Image.Image:
    """
    Crée une heatmap simple (grayscale) depuis une erreur pixel-wise.
    err_map: (28,28) float
    On normalise en [0,255] pour visualiser.
    """
    m = float(err_map.min())
    M = float(err_map.max())
    denom = (M - m) if (M - m) > 1e-12 else 1.0
    norm = ((err_map - m) / denom * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(norm, mode="L")

@torch.no_grad()
def predict(pil_img: Image.Image, add_noise: bool = True, noise_level: float = 0.4):
    """
    Pipeline complet d'inférence:
    1) preprocess image -> x
    2) (option) ajouter bruit -> x_in
    3) reconstruction x_hat + latent z
    4) score anomalie (MSE) + heatmap
    5) décision anomalie (score > threshold)

    Retourne un dict JSON-friendly avec des images en base64.
    """
    model, cfg = load_artifacts()

    x = preprocess_image(pil_img).to(DEVICE)  # image propre (référence)
    x_in = x

    # Pour démo denoising: on ajoute du bruit au moment de l'inférence
    if add_noise:
        x_in = (x + noise_level * torch.randn_like(x)).clamp(0.0, 1.0)

    x_hat, z = model(x_in)

    # Erreur reconstruction pixel-wise
    err = (x - x_hat) ** 2

    # Score d'anomalie = moyenne MSE sur tous pixels
    score = float(err.mean().item())

    # Heatmap: carte 2D (28,28)
    err_map = err.squeeze().detach().cpu().numpy()
    heatmap = make_heatmap(err_map)

    threshold = float(cfg["threshold"])
    is_anomaly = bool(score > threshold)

    return {
        # images
        "original_b64": pil_to_base64(tensor_to_pil(x)),
        "input_b64": pil_to_base64(tensor_to_pil(x_in)),         # bruitée si add_noise=True
        "reconstruction_b64": pil_to_base64(tensor_to_pil(x_hat)),
        "heatmap_b64": pil_to_base64(heatmap),

        # infos
        "score": score,
        "threshold": threshold,
        "is_anomaly": is_anomaly,
        "latent_shape": list(z.shape),  # utile pour afficher "compression"
    }