"""
ImpellerVision — Faz 3: FastAPI Backend
=======================================
Endpoint'ler:
    GET  /health   -> servis + model durumu
    POST /predict  -> goruntu yukle: karar + guven + Grad-CAM isi haritasi (base64)
    POST /track    -> ziyaret bildirimi (Gmail SMTP ile e-posta)
    GET  /         -> frontend (app/static/index.html)

/predict form alanlari:
    file        : goruntu (zorunlu)
    brightness  : float (varsayilan 1.0) — aydinlatma demosu
    contrast    : float (varsayilan 1.0) — aydinlatma demosu
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageEnhance

# src/ modulunu import path'ine ekle
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# .env (gitignore'lu) varsa ortam degiskenlerini yukle — SMTP bilgileri vb.
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from inference import ImpellerPredictor  # noqa: E402
import notifier  # noqa: E402

STATIC_DIR = ROOT / "app" / "static"
MODEL_PATH = ROOT / "models" / "impeller_model.pt"

app = FastAPI(title="ImpellerVision", description="Impeller yüzey kusuru tespiti")

predictor: ImpellerPredictor | None = None


@app.on_event("startup")
def load_model():
    global predictor
    if MODEL_PATH.exists():
        predictor = ImpellerPredictor(MODEL_PATH)
        print(f"[model] yuklendi — cihaz={predictor.device} acc={getattr(predictor,'classes',None)}")
    else:
        print(f"[uyari] model bulunamadi: {MODEL_PATH} — once src/train.py calistirin")


def _to_base64_png(img) -> str:
    """PIL Image ya da np.uint8 HxWx3 diziyi base64 PNG data-URI'ye cevirir."""
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/health")
def health():
    ok = predictor is not None
    return {
        "status": "ok" if ok else "model_yok",
        "model_loaded": ok,
        "device": str(predictor.device) if ok else None,
        "classes": predictor.classes if ok else None,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    brightness: float = Form(1.0),
    contrast: float = Form(1.0),
):
    if predictor is None:
        raise HTTPException(503, "Model yuklenmedi. Once src/train.py ile egitin.")
    try:
        raw = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Gecersiz goruntu dosyasi.")

    # Aydinlatma demosu: parlaklik/kontrast bozulmasi
    if brightness != 1.0:
        image = ImageEnhance.Brightness(image).enhance(brightness)
    if contrast != 1.0:
        image = ImageEnhance.Contrast(image).enhance(contrast)

    result = predictor.predict_with_cam(image, target="defect")

    # Gosterilen girdi — isi haritasiyla ayni cozunurluk (hizalama icin)
    shown = predictor.loc_resize(image)
    return JSONResponse({
        "decision": result["decision"],
        "label": result["label"],
        "confidence": round(result["confidence"] * 100, 2),
        "defect_prob": round(result["defect_prob"] * 100, 2),
        "ok_prob": round(result["ok_prob"] * 100, 2),
        "input_image": _to_base64_png(shown),
        "heatmap": _to_base64_png(result["overlay"]),
        "bbox": result.get("bbox"),          # [x,y,w,h] loc_size koordinatlari / None
        "loc_size": result.get("loc_size"),
        "brightness": brightness,
        "contrast": contrast,
    })


@app.post("/track")
async def track(request: Request, background: BackgroundTasks):
    """Ziyaret bildirimi — frontend sayfa acilisinda bir kez cagirir.
    SMTP ayarli degilse sessizce no-op. E-posta arka planda gonderilir."""
    if not notifier.is_configured():
        return {"ok": False, "reason": "not_configured"}
    # Reverse proxy (nginx/Cloudflare) arkasinda gercek IP X-Forwarded-For'da olur
    xff = request.headers.get("x-forwarded-for")
    ip = xff.split(",")[0].strip() if xff else (request.client.host if request.client else "?")
    # Dedupe anahtari: tarayici ID'si (her kullanici icin omur boyu tek e-posta);
    # ID yoksa IP'ye dus.
    vid = request.query_params.get("vid")
    key = vid or ip
    if notifier.should_send(key):
        background.add_task(
            notifier.send_visit_notification,
            ip,
            request.headers.get("user-agent", "?"),
            request.headers.get("referer", "-"),
            "/",
        )
    return {"ok": True}


# Frontend (en sonda mount edilir ki /health, /predict, /track golgelenmesin)
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
