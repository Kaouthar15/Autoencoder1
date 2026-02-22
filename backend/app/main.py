from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from app.inference import predict

app = FastAPI(title="CDAE API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/reconstruct")
async def reconstruct(file: UploadFile = File(...)):
    content = await file.read()
    img = Image.open(io.BytesIO(content))

    # Denoising demo: bruit ON
    return predict(img, add_noise=True, noise_level=0.4)