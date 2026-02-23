import { useMemo, useState } from "react";
import "./App.css";

type ApiResponse = {
  original_b64: string;
  input_b64?: string;
  reconstruction_b64: string;
  heatmap_b64: string;
  score: number;
  threshold: number;
  is_anomaly: boolean;
  latent_shape?: number[];
};

function b64ToImgSrc(b64: string) {
  return `data:image/png;base64,${b64}`;
}

function formatNum(x: number, digits = 6) {
  return Number.isFinite(x) ? x.toFixed(digits) : String(x);
}

// Fonction pour ajouter du bruit gaussien à une image base64
// (pour montrer visuellement l'effet du dénoising côté frontend)
function addNoiseToImage(b64: string, noiseStrength: number = 0.5): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    
    img.onload = () => {
      try {
        // Créer un canvas
        const canvas = document.createElement("canvas");
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext("2d");
        if (!ctx) {
          resolve(b64); // fallback si canvas ne fonctionne pas
          return;
        }

        // Dessiner l'image
        ctx.drawImage(img, 0, 0);

        // Récupérer les données pixel
        const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imgData.data;

        // Ajouter du bruit gaussien à chaque pixel
        for (let i = 0; i < data.length; i += 4) {
          // Box-Muller pour générer une gaussienne
          const u1 = Math.random();
          const u2 = Math.random();
          const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
          const noise = z * noiseStrength * 255;

          // Ajouter le bruit à RGB
          data[i] = Math.max(0, Math.min(255, data[i] + noise));     // R
          data[i + 1] = Math.max(0, Math.min(255, data[i + 1] + noise)); // G
          data[i + 2] = Math.max(0, Math.min(255, data[i + 2] + noise)); // B
        }

        // Mettre à jour le canvas avec les données bruitées
        ctx.putImageData(imgData, 0, 0);

        // Convertir en data URL (déjà au format data:image/png;base64,...)
        const noisyDataUrl = canvas.toDataURL("image/png");
        resolve(noisyDataUrl);
      } catch (error) {
        console.error("Erreur lors de l'ajout de bruit:", error);
        resolve(b64); // fallback sur l'image originale
      }
    };
    
    img.onerror = () => {
      console.error("Impossible de charger l'image pour ajouter du bruit");
      resolve(b64); // fallback
    };
    
    img.src = b64;
  });
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<ApiResponse | null>(null);
  const [noisyImageB64, setNoisyImageB64] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const fileLabel = useMemo(() => {
    if (!file) return "No file selected";
    return `${file.name} • ${(file.size / 1024).toFixed(1)} KB`;
  }, [file]);

  async function run() {
    if (!file) return;

    setLoading(true);
    setErr(null);
    setResult(null);
    setNoisyImageB64(null);

    const form = new FormData();
    form.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/api/reconstruct", {
        method: "POST",
        body: form,
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const json = (await res.json()) as ApiResponse;
      setResult(json);

      // Générer l'image bruitée côté frontend pour montrer le dénoising
      const noisyB64 = await addNoiseToImage(b64ToImgSrc(json.original_b64), 0.5);
      setNoisyImageB64(noisyB64);
    } catch (e: any) {
      setErr(e?.message ?? "Erreur inconnue");
    } finally {
      setLoading(false);
    }
  }

  function onPick(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0] ?? null;
    setFile(f);
  }

  function reset() {
    setFile(null);
    setResult(null);
    setNoisyImageB64(null);
    setErr(null);
  }

  return (
    <div className="container">
      {/* Header */}
      <div className="header">
        <div className="brand">
          <h1 className="title">CDAE Lab — Denoising & Anomaly Detection</h1>
          <p className="subtitle">
            Upload an image → we preprocess to <b>28×28 grayscale</b>, add noise
            (denoising demo), reconstruct with a <b>Convolutional Denoising Autoencoder</b>,
            then compute a reconstruction-error anomaly score.
          </p>
        </div>

        
      </div>

      {/* Top grid */}
      <div className="gridTop">
        {/* Upload Card */}
        <div className="card">
          <div className="cardHeader">
            <div className="cardTitle">
              <h2>1) Upload</h2>
              <p>Choose any image (we will resize to 28×28).</p>
            </div>
          </div>

          <div className="cardBody">
            <div className="dropzone">
              <div className="dropIcon">⬆️</div>
              <div className="dropText">
                <strong>Upload an image file</strong>
                <span>PNG / JPG • Recommended: simple centered object</span>
              </div>
            </div>

            <div className="fileRow">
              <input type="file" accept="image/*" onChange={onPick} />
              <span className="fileName">{fileLabel}</span>

              <button className="btn" onClick={run} disabled={!file || loading}>
                {loading ? "Running…" : "Run Model"}
              </button>

              <button className="btnSecondary" onClick={reset} disabled={loading}>
                Reset
              </button>
            </div>

            {err && <div className="err">Erreur: {err}</div>}
          </div>
        </div>

        {/* Metrics Card */}
        <div className="card">
          <div className="cardHeader">
            <div className="cardTitle">
              <h2>2) Metrics</h2>
              <p>Reconstruction error + threshold decision.</p>
            </div>

            {result && (
              <div className={"badge " + (result.is_anomaly ? "badgeAnomaly" : "badgeNormal")}>
                {result.is_anomaly ? "Anomaly" : "Normal"}
              </div>
            )}
          </div>

          <div className="cardBody">
            <div className="kpis">
              <div className="kpi">
                <div className="kpiLabel">Score (MSE)</div>
                <div className="kpiValue">{result ? formatNum(result.score) : "—"}</div>
              </div>

              <div className="kpi">
                <div className="kpiLabel">Threshold</div>
                <div className="kpiValue">{result ? formatNum(result.threshold) : "—"}</div>
              </div>

              <div className="kpi">
                <div className="kpiLabel">Latent Shape</div>
                <div className="kpiValue">
                  {result?.latent_shape ? `[${result.latent_shape.join(", ")}]` : "—"}
                </div>
              </div>
            </div>

            <div style={{ marginTop: 12, color: "rgba(255,255,255,0.65)", fontSize: 12 }}>
              <div><b>MSE Score:</b> Mean Squared Error measuring reconstruction quality.</div>
              <div><b>Threshold:</b> Decision boundary calibrated on normal training data (p98).</div>
              <div><b>Latent Shape:</b> Internal compressed feature representation (encoder output).</div>
            </div>


          </div>
        </div>
      </div>

      {/* Results Card (always visible to avoid empty/messy space) */}
      <div className="card" style={{ marginTop: 16 }}>
        <div className="cardHeader">
          <div className="cardTitle">
            <h2>3) Visual Results</h2>
            <p>Original → Noisy input → Reconstruction → Error heatmap</p>
          </div>

          {loading && <div className="badge">Running…</div>}
        </div>

        <div className="cardBody">
          {!result && !loading && (
            <div className="placeholderNote">
              No results yet. Upload an image and click <b>Run Model</b> to display the outputs.
            </div>
          )}

          <div className="resultsGrid">
            {/* Original */}
            <div className="imgCard">
              <h3>Original</h3>
              <div className={"imgWrap " + (!result ? "skeleton" : "")}>
                {result ? (
                  <img src={b64ToImgSrc(result.original_b64)} alt="Original" />
                ) : (
                  <div className="imgPlaceholder">Original image will appear here</div>
                )}
              </div>
            </div>

            {/* Input (Noisy) */}
            <div className="imgCard">
              <h3>Input (Noisy)</h3>
              <div className={"imgWrap " + (!result ? "skeleton" : "")}>
                {noisyImageB64 ? (
                  <img src={noisyImageB64} alt="Noisy input" />
                ) : (
                  <div className="imgPlaceholder">
                    Noisy input will appear here
                    <br />
                    <span style={{ opacity: 0.8 }}></span>
                  </div>
                )}
              </div>
            </div>

            {/* Reconstruction */}
            <div className="imgCard">
              <h3>Reconstruction</h3>
              <div className={"imgWrap " + (!result ? "skeleton" : "")}>
                {result ? (
                  <img src={b64ToImgSrc(result.reconstruction_b64)} alt="Reconstruction" />
                ) : (
                  <div className="imgPlaceholder">Reconstruction will appear here</div>
                )}
              </div>
            </div>

            {/* Heatmap */}
            <div className="imgCard">
              <h3>Error Heatmap</h3>
              <div className={"imgWrap " + (!result ? "skeleton" : "")}>
                {result ? (
                  <img src={b64ToImgSrc(result.heatmap_b64)} alt="Error heatmap" />
                ) : (
                  <div className="imgPlaceholder">Error heatmap will appear here</div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
