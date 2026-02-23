import os
import json
import io
import base64
import numpy as np
import torch
from PIL import Image

from app.model import CDAE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_cfg = None


def load_artifacts():
    """
    Charge 1 seule fois :
    - le modèle (cdae.pt)
    - la config threshold.json
    """
    global _model, _cfg

    if _model is not None and _cfg is not None:
        return _model, _cfg

    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    weights_path = os.path.join(assets_dir, "cdae.pt")
    cfg_path = os.path.join(assets_dir, "threshold.json")

    if not os.path.exists(weights_path) or not os.path.exists(cfg_path):
        raise FileNotFoundError(
            "Poids/config manquants. Lance d'abord : python -m app.train"
        )

    with open(cfg_path, "r", encoding="utf-8") as f:
        _cfg = json.load(f)

    _model = CDAE(latent_channels=int(_cfg["latent_channels"])).to(DEVICE)
    state = torch.load(weights_path, map_location=DEVICE)
    _model.load_state_dict(state)
    _model.eval()

    return _model, _cfg


def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    """
    Identique au training Fashion-MNIST :
    - grayscale
    - resize 28x28
    - normalisation [0,1]
    - shape (1,1,28,28)
    """
    img = pil_img.convert("L").resize((28, 28))
    arr = np.array(img).astype(np.float32) / 255.0
    x = torch.from_numpy(arr)[None, None, :, :]
    return x


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    arr = (x.squeeze().detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def make_heatmap(err_map: np.ndarray) -> Image.Image:
    m = float(err_map.min())
    M = float(err_map.max())
    denom = (M - m) if (M - m) > 1e-12 else 1.0
    norm = ((err_map - m) / denom * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(norm, mode="L")


@torch.no_grad()
def predict(pil_img: Image.Image, add_noise: bool = True):
    """
    Pipeline STRICTEMENT identique au training :

    1) preprocess -> x_clean
    2) ajout bruit (même niveau que training)
    3) reconstruction
    4) score = mean((x_clean - x_hat)^2)
    5) comparaison au threshold
    """

    model, cfg = load_artifacts()

    x = preprocess_image(pil_img).to(DEVICE)

    # 🔥 IMPORTANT : utiliser le même bruit que training
    noise_level = float(cfg.get("noise", 0.0))

    x_in = x
    if add_noise:
        x_in = (x + noise_level * torch.randn_like(x)).clamp(0.0, 1.0)

    x_hat, z = model(x_in)

    # Score IDENTIQUE au training
    err = (x - x_hat) ** 2
    score = float(err.mean().item())

    threshold = float(cfg["threshold"])
    is_anomaly = bool(score > threshold)

    # Debug utile (à retirer si prod)
    print("----- DEBUG INFERENCE -----")
    print("Noise level:", noise_level)
    print("Threshold:", threshold)
    print("Score:", score)
    print("Is anomaly:", is_anomaly)
    print("----------------------------")

    err_map = err.squeeze().detach().cpu().numpy()
    heatmap = make_heatmap(err_map)

    return {
        "original_b64": pil_to_base64(tensor_to_pil(x)),
        "input_b64": pil_to_base64(tensor_to_pil(x_in)),
        "reconstruction_b64": pil_to_base64(tensor_to_pil(x_hat)),
        "heatmap_b64": pil_to_base64(heatmap),

        "score": score,
        "threshold": threshold,
        "is_anomaly": is_anomaly,
        "latent_shape": list(z.shape),
    }