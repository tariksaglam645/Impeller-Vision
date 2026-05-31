"""
ImpellerVision — Ortak Cikarim Modulu
=====================================
Hem Grad-CAM script'i (Faz 2) hem FastAPI backend (Faz 3) bu modulu kullanir.
Tek kaynak: model yukleme + tahmin + isi haritasi lokalizasyonu.

Lokalizasyon hassasiyeti icin:
  * Karar (PASS/FAIL) egitim cozunurlugunde (224) hesaplanir -> guvenilir kalir.
  * Isi haritasi AYRI ve daha yuksek cozunurlukte (varsayilan 320) + orta seviye
    katmandan (stride ~16) + LayerCAM ile hesaplanir -> cok daha keskin harita.
  * FAIL durumunda isi haritasi esiklenip kusur bolgesine bounding-box cizilir.

Kayit formati (models/impeller_model.pt):
    {arch, img_size, classes, class_to_idx, defect_index, normalize, state_dict}
"""
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT / "models" / "impeller_model.pt"

_CAM_METHODS = {"gradcam": GradCAM, "gradcam++": GradCAMPlusPlus, "layercam": LayerCAM}


def _build_model(arch: str, num_classes: int) -> torch.nn.Module:
    if arch != "mobilenet_v2":
        raise ValueError(f"Desteklenmeyen mimari: {arch}")
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    return model


class ImpellerPredictor:
    """Impeller kusur tahmini + yuksek cozunurluklu isi haritasi lokalizasyonu."""

    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        device: Optional[str] = None,
        cam_method: str = "layercam",
        loc_size: int = 320,
        target_stride: int = 16,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        self.classes = ckpt["classes"]                 # ['def_front', 'ok_front']
        self.defect_index = ckpt["defect_index"]
        self.ok_index = 1 - self.defect_index
        self.img_size = ckpt["img_size"]               # 224 — karar cozunurlugu
        self.loc_size = loc_size                       # isi haritasi cozunurlugu
        self.mean = ckpt["normalize"]["mean"]
        self.std = ckpt["normalize"]["std"]

        self.model = _build_model(ckpt["arch"], len(self.classes)).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        # Daha ince harita icin orta seviye (stride ~16) katmani hedefle.
        self.target_layer, self.target_idx, self.cam_grid = self._pick_target_layer(target_stride)
        cam_cls = _CAM_METHODS[cam_method.lower()]
        self.cam_method = cam_method.lower()
        self._cam = cam_cls(model=self.model, target_layers=[self.target_layer])

        self.normalize = transforms.Normalize(self.mean, self.std)
        self.resize = transforms.Resize((self.img_size, self.img_size))   # 224
        self.loc_resize = transforms.Resize((self.loc_size, self.loc_size))

    # -- hedef katman secimi -------------------------------------------------
    def _pick_target_layer(self, target_stride: int):
        """features icinde, uzamsal boyutu loc_size/target_stride'tan kucuk olmayan
        EN DERIN katmani sec (semantik + yeterince ince)."""
        target = self.loc_size / target_stride
        chosen_idx = len(self.model.features) - 1
        x = torch.zeros(1, 3, self.loc_size, self.loc_size, device=self.device)
        with torch.no_grad():
            h = x
            for i, layer in enumerate(self.model.features):
                h = layer(h)
                if h.shape[-1] >= target:
                    chosen_idx = i
        # secilen katmanin uzamsal boyutu
        with torch.no_grad():
            h = x
            for i, layer in enumerate(self.model.features):
                h = layer(h)
                if i == chosen_idx:
                    grid = h.shape[-1]
                    break
        return self.model.features[chosen_idx], chosen_idx, grid

    # -- on isleme -----------------------------------------------------------
    def _rgb_float(self, image: Image.Image, size_resize) -> np.ndarray:
        img = size_resize(image.convert("RGB"))
        return np.asarray(img, dtype=np.float32) / 255.0

    def _to_tensor(self, rgb_float: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(rgb_float).permute(2, 0, 1)  # HWC -> CHW
        t = self.normalize(t)
        return t.unsqueeze(0).to(self.device)

    # -- tahmin (224 — guvenilir karar) --------------------------------------
    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        rgb = self._rgb_float(image, self.resize)
        tensor = self._to_tensor(rgb)
        probs = F.softmax(self.model(tensor), dim=1)[0].cpu().numpy()
        defect_prob = float(probs[self.defect_index])
        ok_prob = float(probs[self.ok_index])
        is_defect = defect_prob >= 0.5
        return {
            "decision": "FAIL" if is_defect else "PASS",
            "label": "Defect" if is_defect else "OK",
            "confidence": defect_prob if is_defect else ok_prob,
            "defect_prob": defect_prob,
            "ok_prob": ok_prob,
        }

    # -- bounding box --------------------------------------------------------
    @staticmethod
    def _cam_bbox(cam: np.ndarray, rel_thresh: float = 0.5, min_area_frac: float = 0.002):
        """Isi haritasini esikle, en buyuk bolgenin kutusunu dondur (x,y,w,h) / None."""
        mask = (cam >= rel_thresh * cam.max()).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < min_area_frac * cam.size:
            return None
        x, y, w, h = cv2.boundingRect(c)
        return [int(x), int(y), int(w), int(h)]

    # -- isi haritasi (yuksek cozunurluk) ------------------------------------
    def predict_with_cam(self, image: Image.Image, target: str = "defect", draw_box: bool = True) -> dict:
        """Tahmin + yuksek cozunurluklu isi haritasi overlay (uint8 RGB) + bbox.

        Karar 224'te; harita loc_size'da. target='defect' -> her zaman kusur kaniti.
        """
        pred = self.predict(image)  # guvenilir karar (224)

        if target == "pred":
            target_idx = self.defect_index if pred["decision"] == "FAIL" else self.ok_index
        else:
            target_idx = self.defect_index

        rgb = self._rgb_float(image, self.loc_resize)   # yuksek cozunurluk
        tensor = self._to_tensor(rgb)
        grayscale_cam = self._cam(
            input_tensor=tensor,
            targets=[ClassifierOutputTarget(target_idx)],
        )[0]
        overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)  # uint8 RGB

        # Kusur kutusu — yalniz FAIL'de anlamli (PASS'te kusur yok)
        bbox = None
        if draw_box and pred["decision"] == "FAIL":
            bbox = self._cam_bbox(grayscale_cam)
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 2)

        pred["overlay"] = overlay              # np.uint8 HxWx3 (loc_size)
        pred["cam"] = grayscale_cam            # float [0,1]
        pred["bbox"] = bbox                    # [x,y,w,h] loc_size koordinatlari / None
        pred["loc_size"] = self.loc_size
        return pred
