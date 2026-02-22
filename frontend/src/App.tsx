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

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<ApiResponse | null>(null);
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

        <div className="pills">
          <div className="pill">Dataset: Fashion-MNIST</div>
          <div className="pill">Backend: FastAPI + PyTorch</div>
          <div className="pill">Latent: 16×4×4</div>
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
              Tip: test with Fashion-MNIST samples (label 0 vs others) to see a clearer
              Normal/Anomaly separation.
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
                {result?.input_b64 ? (
                  <img src={b64ToImgSrc(result.input_b64)} alt="Noisy input" />
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