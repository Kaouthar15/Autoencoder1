"""Mini-API pour tester la reconstruction depuis le navigateur.

Je garde tout minimal : un healthcheck et un endpoint `reconstruct` qui accepte
un fichier via `FormData` (ce que `fetch` / `FormData.append('file', file)` envoie).
Si tu changes le port du front (vite), pense à mettre à jour `allow_origins`.
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from app.inference import predict

app = FastAPI(title="CDAE API")

app.add_middleware(
        CORSMiddleware,
        # Si tu utilises le dev server Vite, il tourne souvent sur 5173.
        # Ajuste ou ajoute des origins si nécessaire.
        allow_origins=["http://localhost:5174"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/reconstruct")
async def reconstruct(file: UploadFile = File(...)):
        """Reçoit une image uploadée et renvoie le dict produit par `predict()`.

        Utilise `UploadFile` car c'est pratique avec `fetch` et `FormData` côté front.
        """
        content = await file.read()
        img     = Image.open(io.BytesIO(content))

        # On envoie l'image propre au modèle — pas de bruit ajouté ici.
        return predict(img)