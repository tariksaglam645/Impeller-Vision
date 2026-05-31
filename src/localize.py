"""
ImpellerVision — Faz 2: Lokalizasyon Ornekleri
==============================================
Egitilmis modeli kullanarak test setinden ornek goruntuler uzerinde isi haritasi
(varsayilan: LayerCAM) + bounding-box uretir. Cikti: outputs/localization_samples/

Kullanilan CAM yontemi src/inference.py icinde secilir (gradcam / gradcam++ / layercam).

Her ornek panelinde: orijinal | isi haritasi overlay (karar + guven skoru basligi).

Kullanim:
    python src/localize.py                # her siniftan 6 ornek
    python src/localize.py --n 8
"""
import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from inference import ImpellerPredictor, ROOT

TEST_DIR = ROOT / "data" / "test"
OUT_DIR = ROOT / "outputs" / "localization_samples"


def list_images(folder: Path, n: int, seed: int = 42):
    files = sorted([p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    random.Random(seed).shuffle(files)
    return files[:n]


def make_panel(predictor, img_path: Path, out_path: Path):
    image = Image.open(img_path).convert("RGB")
    result = predictor.predict_with_cam(image, target="defect")

    orig = np.asarray(predictor.resize(image), dtype=np.uint8)
    overlay = result["overlay"]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.8))
    axes[0].imshow(orig); axes[0].set_title("Orijinal"); axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(f"{predictor.cam_method} (kusur kaniti + kutu)"); axes[1].axis("off")

    color = "#c0392b" if result["decision"] == "FAIL" else "#27ae60"
    fig.suptitle(
        f"{img_path.name}  →  {result['decision']} "
        f"({result['label']}, güven {result['confidence']*100:.1f}%)",
        color=color, fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return result["decision"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6, help="her siniftan ornek sayisi")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    predictor = ImpellerPredictor()
    print(f"[cihaz] {predictor.device} | siniflar {predictor.classes}")

    for cls in ("def_front", "ok_front"):
        imgs = list_images(TEST_DIR / cls, args.n)
        for p in imgs:
            out = OUT_DIR / f"{cls}__{p.stem}.png"
            decision = make_panel(predictor, p, out)
            print(f"  {cls}/{p.name:20} -> {decision}  ({out.name})")

    print(f"\n[kayit] Lokalizasyon ornekleri -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
