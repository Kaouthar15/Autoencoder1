"""
Ici on garde les choses simples et explicites :
- on charge le modèle et la config une seule fois (c'est plus rapide pour l'API),
- on n'ajoute jamais de bruit côté inference (c'est cohérent avec le threshold),
- on renvoie des images encodées en base64 pour que le front les affiche facilement.
"""

import os
import json
import io
import base64
import numpy as np
import torch
from PIL import Image

from app.model import CDAE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Module-level cache: on charge le modèle et la configuration une seule fois.
# Ceci évite de relire les fichiers depuis le disque à chaque requête HTTP.
_model = None
_cfg   = None


def load_artifacts():
    """Charge le modèle et la config depuis `app/assets` la première fois.

    Petit mode d'emploi : si les fichiers n'existent pas, on lève une erreur claire
    pour te dire de lancer d'abord l'entraînement. Le modèle est chargé sur le
    `DEVICE` détecté et mis en `eval()`.
    """
    global _model, _cfg

    if _model is not None and _cfg is not None:
        return _model, _cfg

    assets_dir    = os.path.join(os.path.dirname(__file__), "assets")
    weights_path  = os.path.join(assets_dir, "cdae.pt")
    cfg_path      = os.path.join(assets_dir, "threshold.json")

    if not os.path.exists(weights_path) or not os.path.exists(cfg_path):
        raise FileNotFoundError(
            "Poids/config manquants. Lance d'abord : python -m app.train"
        )

    with open(cfg_path, "r", encoding="utf-8") as f:
        _cfg = json.load(f)

    _model = CDAE(latent_channels=int(_cfg["latent_channels"])).to(DEVICE)
    state  = torch.load(weights_path, map_location=DEVICE)
    _model.load_state_dict(state)
    _model.eval()

    return _model, _cfg


def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    """Prépare l'image pour le modèle :

    - conversion en niveaux de gris
    - redimensionnement en 28×28
    - mise à l'échelle dans [0, 1]
    - ajout des dimensions batch/channel pour obtenir (1,1,28,28)

    On garde exactement le même preprocess que pendant l'entraînement pour
    éviter toute surprise dans la distribution des données.
    """
    img = pil_img.convert("L").resize((28, 28))
    arr = np.array(img).astype(np.float32) / 255.0
    x   = torch.from_numpy(arr)[None, None, :, :]
    return x


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    arr = (x.squeeze().detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def make_heatmap(err_map: np.ndarray) -> Image.Image:
    m     = float(err_map.min())
    M     = float(err_map.max())
    denom = (M - m) if (M - m) > 1e-12 else 1.0
    norm  = ((err_map - m) / denom * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(norm, mode="L")


@torch.no_grad()
def predict(pil_img: Image.Image):
    """Fait la reconstruction et renvoie un dict prêt à être sérialisé.

    Étapes : preprocessing → reconstruction (pas de bruit) → calcul du score
    → comparaison au threshold. Le dict contient aussi des images encodées en
    base64 pour un affichage direct côté frontend.
    """
    model, cfg = load_artifacts()

    # Image propre uniquement — pas de bruit en inférence (cohérent avec le seuil)
    x     = preprocess_image(pil_img).to(DEVICE)
    x_hat, z = model(x)

    # Score MSE
    err   = (x - x_hat) ** 2
    score = float(err.mean().item())

    threshold  = float(cfg["threshold"])
    is_anomaly = bool(score > threshold)

    print("----- DEBUG INFERENCE -----")
    print(f"Score     : {score:.6f}")
    print(f"Threshold : {threshold:.6f}")
    print(f"Anomaly   : {is_anomaly}")
    print("----------------------------")

    err_map = err.squeeze().detach().cpu().numpy()
    heatmap = make_heatmap(err_map)

    # x_in affiché = image propre (pas de bruit ajouté côté inférence)
    return {
        "original_b64":       pil_to_base64(tensor_to_pil(x)),
        "input_b64":          pil_to_base64(tensor_to_pil(x)),
        "reconstruction_b64": pil_to_base64(tensor_to_pil(x_hat)),
        "heatmap_b64":        pil_to_base64(heatmap),
        "score":              score,
        "threshold":          threshold,
        "is_anomaly":         is_anomaly,
        "latent_shape":       list(z.shape),
    }